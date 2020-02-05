"""
Code for training via knowledge distillation on the CIFAR10 dataset.
"""
import torch
from torch import optim
from torchvision.models import vgg16_bn

from networks.student import StudentNet
from utils.checkpoints import load_model_from_checkpoint
from utils.init import initialize
from utils.gpu_utils import get_gpu_if_available
from utils.logs import save_dict_as_json
from model_trainers.distillation_trainer import KnowledgeDistillationModelTrainer
from utils.options import knowledge_distillation_options


def warm_up(epoch: int) -> float:  # Epochs are zero-indexed in schedulers by default. Different from my convention.
    if epoch < 5:  # Linear warm-up to ease learning.
        return (epoch + 1) / 5
    elif epoch < 300:
        return 1.
    elif epoch < 350:
        return 0.1
    elif epoch < 375:
        return 0.01
    else:
        return 0.001


def main(opt):
    run_number, run_name, log_path, checkpoint_path = initialize(opt.record_path, train_method=opt.train_method)
    device = get_gpu_if_available(gpu=opt.gpu)
    print(f'Using device {device}.')

    # The teacher and student model types and settings are fixed by the task.
    # While best to set it with options, that would make it too hard to read.
    student = StudentNet(in_channels=3, num_classes=10).to(device, non_blocking=True)
    teacher = vgg16_bn(num_classes=10)  # Settings for CIFAR10 dataset.
    load_model_from_checkpoint(model=teacher, load_dir=opt.teacher_checkpoint)
    teacher = teacher.to(device, non_blocking=True)

    optimizer = optim.SGD(params=student.parameters(), lr=opt.lr, momentum=opt.momentum,
                          weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up)

    # Saving options and extra information in json file for later viewing.
    opt.run_name = run_name
    opt.device = str(device)
    opt.train_method = 'Knowledge Distillation'
    opt.student_name = student.__class__.__name__
    opt.teacher_name = teacher.__class__.__name__
    opt.optimizer_name = optimizer.__class__.__name__
    opt.scheduler_name = scheduler.__class__.__name__
    save_dict_as_json(vars(opt), log_dir=log_path, save_name='knowledge_distillation_options')

    trainer = KnowledgeDistillationModelTrainer(
        teacher=teacher, student=student, optimizer=optimizer, scheduler=scheduler, dataset='CIFAR10',
        batch_size=opt.batch_size, num_workers=opt.num_workers, distill_ratio=opt.distill_ratio,
        temperature=opt.temperature, data_path=opt.data_path, log_path=log_path, checkpoint_path=checkpoint_path)

    trainer.train_model(num_epochs=opt.num_epochs)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # Increase speed if input sizes are the same.
    options = dict(
        teacher_checkpoint='../records/Teacher/Trial 08  2020-01-30 14-22-32/checkpoints/checkpoint_040.tar',
        num_workers=2,
        lr=0.1,
        distill_ratio=0.95,
        temperature=1.,
        gpu=0
    )
    options = knowledge_distillation_options(**options).parse_args()
    main(options)
