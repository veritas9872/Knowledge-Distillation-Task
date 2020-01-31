from time import time
from typing import Union
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter

from train.get_loaders import get_cifar10_loaders
from train.kd_loss import KnowledgeDistillationLoss
from utils.checkpoints import CheckpointManager
from utils.logs import get_logger
from utils.gpu_utils import get_single_model_device


class KnowledgeDistillationModelTrainer:
    SchedulerType = Union[optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau]

    def __init__(self, teacher: nn.Module, student: nn.Module, optimizer: optim.Optimizer, scheduler: SchedulerType,
                 dataset: str, batch_size: int, num_workers: int, distill_ratio: float, temperature: float,
                 data_path: Path, log_path: Path, checkpoint_path: Path):
        self.logger = get_logger(name=__name__, save_dir=str(log_path / 'logs'))
        self.logger.info('Initializing Knowledge Distillation Model Trainer.')

        assert get_single_model_device(teacher) == get_single_model_device(student), \
            'Teacher and student are expected to be on the same single device.'

        if dataset.upper() == 'CIFAR10':
            train_loader, eval_loader = get_cifar10_loaders(
                data_root=str(data_path), batch_size=batch_size, num_workers=num_workers, augment=True)
        else:
            raise NotImplementedError('Only CIFAR10 implemented.')

        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.device = get_single_model_device(student)  # Gets device of module if on a single device.
        self.loss_func = KnowledgeDistillationLoss(distill_ratio=distill_ratio, temperature=temperature)
        self.writer = SummaryWriter(str(log_path))
        self.manager = CheckpointManager(student, optimizer, checkpoint_path, mode='max', save_best_only=True)
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epoch = 0
        self.tic = 0
        self.tic_tic = time()

        self.teacher.eval()  # Set teacher to evaluation mode for more consistent output.

    def _train_epoch(self) -> float:
        self.tic = time()
        self.student.train()
        torch.autograd.enable_grad = True  # Allow gradient calculations.
        metrics = defaultdict(list)  # dict of lists to keep track of losses.
        correct = torch.tensor(0, device=self.device)  # Counter for number of correct predictions.

        for inputs, targets in self.train_loader:
            targets = targets.to(self.device)
            inputs = inputs.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.no_grad():  # Remove gradient calculation for teacher to increase speed.
                teacher_logits: Tensor = self.teacher(inputs)  # Teacher should not be training during KD.
            student_logits: Tensor = self.student(inputs)  # ': Tensor' indicates type hinting. No effect on anything.
            kd_loss, loss_dict = self.loss_func(student_logits, teacher_logits, targets)
            kd_loss.backward()
            self.optimizer.step()

            # Getting metrics
            with torch.no_grad():  # Number of correct values. Maximum of outputs is the same as softmax maximum.
                correct += (targets == student_logits.argmax(dim=1)).sum().detach()
            metrics['kd_loss'].append(kd_loss.detach())
            for key, value in loss_dict.items():
                metrics[key].append(value.detach())

        accuracy = correct.item() / len(self.train_loader.dataset) * 100
        self._write_epoch_metrics(accuracy=accuracy, metrics=metrics, is_train=True)
        return accuracy

    def _eval_epoch(self) -> float:
        self.tic = time()
        self.student.eval()
        torch.autograd.enable_grad = False  # Remove gradient calculation for speedup.
        metrics = defaultdict(list)  # dict of lists to keep track of losses.
        correct = torch.tensor(0, device=self.device)  # Counter for number of correct predictions.

        for inputs, targets in self.eval_loader:
            targets = targets.to(self.device)
            inputs = inputs.to(self.device)
            teacher_logits: Tensor = self.teacher(inputs)
            student_logits: Tensor = self.student(inputs)
            kd_loss, loss_dict = self.loss_func(student_logits, teacher_logits, targets)

            correct += (targets == student_logits.argmax(dim=1)).sum().detach()
            metrics['kd_loss'].append(kd_loss.detach())
            for key, value in loss_dict.items():
                metrics[key].append(value.detach())

        accuracy = correct.item() / len(self.eval_loader.dataset) * 100
        self._write_epoch_metrics(accuracy=accuracy, metrics=metrics, is_train=False)
        return accuracy

    def _write_epoch_metrics(self, accuracy: float, metrics: dict, is_train: bool):
        phase = 'Train' if is_train else 'Eval'
        self.logger.info(f'Epoch {self.epoch:02d} {phase} accuracy {accuracy:.1f}%.')
        self.writer.add_scalar(tag=f'{phase}/epoch_accuracy', scalar_value=accuracy, global_step=self.epoch)
        epoch_metrics = dict()
        metric_string = f'Epoch {self.epoch:02d} {phase} '
        for key, value in metrics.items():  # Writing the component losses to Tensorboard.
            epoch_metric = torch.stack(value).mean().item()
            epoch_metrics[key] = epoch_metric
            metric_string += f'{key}: {epoch_metric:.3f}, '
            self.writer.add_scalar(tag=f'{phase}/epoch_{key}', scalar_value=epoch_metric, global_step=self.epoch)
        toc = int(time() - self.tic)
        metric_string += f'Time: {toc}s.'
        self.logger.info(metric_string)

    def _write_learning_rates(self):  # Recording learning rate.
        for idx, group in enumerate(self.optimizer.param_groups, start=1):  # 1-based indexing.
            self.writer.add_scalar(tag=f'Learning Rate {idx}', scalar_value=group['lr'], global_step=self.epoch)

    def _scheduler_step(self, metrics):
        if self.scheduler is not None:  # No learning rate scheduling if scheduler is None.
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=metrics)
            else:
                self.scheduler.step()

    def _train_model(self, num_epochs: int) -> float:
        best_acc = 0.
        for epoch in range(1, num_epochs + 1):  # 1 based indexing. tqdm for progress bar.
            self.epoch = epoch  # Update epoch.
            train_epoch_acc = self._train_epoch()
            eval_epoch_acc = self._eval_epoch()
            best_acc = max(best_acc, eval_epoch_acc)  # Update best performance if
            self._write_learning_rates()
            self.manager.save(metric=eval_epoch_acc)  # Save checkpoint.
            over_fit = train_epoch_acc - eval_epoch_acc  # Positive values indicate over-fitting.
            self.writer.add_scalar(tag='Over-fitting', scalar_value=over_fit, global_step=self.epoch)
            self.logger.info(f'Epoch {self.epoch:02d} Over-fitting: {over_fit:.3f}.')
            self._scheduler_step(metrics=eval_epoch_acc)  # Scheduler step for all scheduler types.
        return best_acc

    def train_model(self, num_epochs: int) -> float:
        try:  # Including safeguards against keyboard interruption.
            best_acc = self._train_model(num_epochs=num_epochs)
            self.writer.flush()
            toc_toc = int(time() - self.tic_tic)
            self.logger.info(f'Finished Training. Best performance: {best_acc:.2f}%. '
                             f'Time: {toc_toc // 60}min {toc_toc % 60}s.')
            return best_acc
        except KeyboardInterrupt:
            self.writer.flush()  # Write to tensorboard before terminating.
            self.logger.info('Training interrupted before completion.')
            return -1



