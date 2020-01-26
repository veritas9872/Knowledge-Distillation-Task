"""
Code for training via knowledge distillation on the CIFAR10 dataset.
"""
from torch import optim
from torchvision.models import vgg16_bn

from networks.student import StudentNet
from utils.checkpoints import load_model_from_checkpoint
from utils.init import initialize, get_gpu_if_available
from utils.logs import save_dict_as_json
from model_trainers.kd_trainer import KnowledgeDistillationModelTrainer
from utils.options import knowledge_distillation_options


def main(opt):
    run_number, run_name, log_path, checkpoint_path = initialize(opt.record_path, train_method='KD')
    device = get_gpu_if_available(gpu=opt.gpu)
    print(f'Using device {device}.')

    # The teacher and student model types and settings are fixed by the task.
    # While best to set it with options, that is too much work.
    student = StudentNet(in_channels=3, num_classes=10).to(device, non_blocking=True)
    teacher = vgg16_bn(num_classes=10)
    load_model_from_checkpoint(model=teacher, load_dir=opt.teacher_checkpoint)
    teacher = teacher.to(device, non_blocking=True)

    optimizer = optim.Adam(params=student.parameters(), lr=opt.lr)
    scheduler = None  # No learning rate scheduling.

    # Saving options and extra information in json file for later viewing.
    opt.run_name = run_name
    opt.device = str(device)
    opt.train_method = 'KD'
    opt.student_name = student.__class__.__name__
    opt.teacher_name = teacher.__class__.__name__
    opt.optimizer_name = optimizer.__class__.__name__
    opt.scheduler_name = scheduler.__class__.__name__
    save_dict_as_json(vars(opt), log_dir=log_path, save_name='knowledge_distillation_options')

    trainer = KnowledgeDistillationModelTrainer(
        teacher=teacher, student=student, optimizer=optimizer, scheduler=scheduler, dataset='CIFAR10',
        batch_size=opt.batch_size, alpha=opt.alpha, temperature=opt.temperature, data_path=opt.data_path,
        log_path=str(log_path), checkpoint_path=str(checkpoint_path))

    trainer.train_model(num_epochs=opt.num_epochs)


if __name__ == '__main__':
    options = knowledge_distillation_options().parse_args()
    main(options)
