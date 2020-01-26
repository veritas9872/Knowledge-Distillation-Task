from typing import Union

import torch
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from train.get_loaders import get_cifar10_loaders
from utils.checkpoints import CheckpointManager
from utils.logs import get_logger


class ClassificationModelTrainer:
    SchedulerType = Union[optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau]

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: SchedulerType,
                 dataset: str, batch_size: int, data_path: str, log_path: str, checkpoint_path: str):
        torch.backends.cudnn.benchmark = True  # Increase speed assuming input sizes are the same, which they are.

        self.logger = get_logger(name=__name__, save_dir=log_path)
        self.logger.info('Initializing Classification Model Trainer.')

        if dataset.upper() == 'CIFAR10':
            train_loader, eval_loader = get_cifar10_loaders(
                data_root=data_path, batch_size=batch_size, num_workers=1)
        else:
            raise NotImplementedError('Only CIFAR10 implemented.')

        self.model = model
        self.optimizer = optimizer
        self.device = model.device
        self.loss_func = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_path)
        self.manager = CheckpointManager(model, optimizer, checkpoint_path, save_best_only=True)
        self.scheduler = scheduler  # No learning rate scheduling if scheduler = None.
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epoch = 0

    def _train_epoch(self) -> float:
        self.model.train()  # Change settings for batchnorm, dropout, etc.
        torch.autograd.enable_grad = True  # Allow gradient calculations.
        losses = list()
        correct = torch.tensor(0, device=self.device)  # Counter for number of correct predictions.

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)  # Asynchronous transfer to minimize data starvation.
            self.optimizer.zero_grad()
            outputs: Tensor = self.model(inputs)  # Type hinting 'outputs'. This does not affect the value in any way.
            loss = self.loss_func(outputs, targets.to(self.device))
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())

            with torch.no_grad:  # Number of correct values. Maximum of outputs is the same as softmax maximum.
                correct += (targets == outputs.argmax(dim=1)).detach()

        accuracy = correct.item() / len(self.train_loader.dataset) * 100
        epoch_loss = torch.mean(torch.stack(losses)).item()  # Minimizing device to host data transfer this way.
        self.writer.add_scalar(tag='Train/epoch_loss', scalar_value=epoch_loss, global_step=self.epoch)
        self.writer.add_scalar(tag='Train/epoch_accuracy', scalar_value=accuracy, global_step=self.epoch)
        self.logger.info(f'Epoch {self.epoch:02d} Train epoch loss: {epoch_loss:.3f}, epoch accuracy {accuracy:.1f}%.')
        for idx, group in enumerate(self.optimizer.param_groups, start=1):  # Recording learning rate.
            self.writer.add_scalar(tag=f'Learning Rate {idx}', scalar_value=group['lr'])

        return accuracy

    def _eval_epoch(self) -> float:
        self.model.eval()
        torch.autograd.enable_grad = False  # Disable gradient calculations for faster calculations.
        losses = list()
        correct = torch.tensor(0, device=self.device)  # Counter for number of correct predictions.

        for inputs, targets in self.eval_loader:
            outputs = self.model(inputs.to(self.device))  # Asynchronous transfer is impossible for evaluation.
            loss = self.loss_func(outputs, targets.to(self.device))
            losses.append(loss)
            correct += (targets == outputs.argmax(dim=1))

        accuracy = correct.item() / len(self.eval_loader.dataset) * 100
        epoch_loss = torch.mean(torch.stack(losses)).item()
        self.writer.add_scalar(tag='Eval/epoch_loss', scalar_value=epoch_loss, global_step=self.epoch)
        self.writer.add_scalar(tag='Eval/epoch_accuracy', scalar_value=accuracy, global_step=self.epoch)
        self.logger.info(f'Epoch {self.epoch:02d} Eval epoch loss: {epoch_loss:.3f} epoch accuracy {accuracy:.1f}%.')

        return accuracy

    def _scheduler_step(self, metrics):
        if self.scheduler is not None:  # No learning rate scheduling if scheduler is None.
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=metrics)
            else:
                self.scheduler.step()

    def _train_model(self, num_epochs: int):
        for epoch in tqdm(range(1, num_epochs + 1)):  # 1 based indexing. tqdm for progress bar.
            self.epoch = epoch
            train_epoch_acc = self._train_epoch()
            eval_epoch_acc = self._eval_epoch()
            self.manager.save(metric=eval_epoch_acc)
            over_fit = eval_epoch_acc - train_epoch_acc
            self.writer.add_scalar(tag='Over-fitting', scalar_value=over_fit, global_step=self.epoch)
            self.logger.info(f'Epoch {self.epoch:02d} Over-fitting: {over_fit:.3f}%.')
            self._scheduler_step(metrics=eval_epoch_acc)  # Scheduler step for all scheduler types.

    def train_model(self, num_epochs: int):
        try:  # Including safeguards against keyboard interruption.
            self._train_model(num_epochs=num_epochs)
            self.logger.info('Finished Training.')
        except KeyboardInterrupt:
            self.writer.flush()  # Write to tensorboard before terminating.
            self.logger.info('Training interrupted before completion.')
