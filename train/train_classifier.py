"""
Code for training the teacher network on the CIFA10 classification task.
"""
from torch import nn, optim

from utils.init import initialize
from utils.gpu_utils import get_gpu_if_available
from utils.logs import save_dict_as_json
from model_trainers.classify_trainer import ClassificationModelTrainer


def warm_up(epoch: int) -> float:  # Epochs are zero-indexed in schedulers by default. Different from my convention.
    if epoch < 5:  # Linear warm-up to ease learning.
        return (epoch+1) / 5
    elif epoch < 300:
        return 1.
    elif epoch < 350:
        return 0.1
    elif epoch < 375:
        return 0.01
    else:
        return 0.001


def train_classifier(opt, model: nn.Module, train_method: str):
    # Initialize the run.
    run_number, run_name, log_path, checkpoint_path = initialize(opt.record_path, train_method=train_method)
    device = get_gpu_if_available(gpu=opt.gpu)
    print(f'Using device {device}.')

    model = model.to(device)
    optimizer = optim.SGD(params=model.parameters(), lr=opt.lr,
                          momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    # scheduler = None
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 350], gamma=0.1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up)

    # Saving options and extra information in json file for later viewing.
    opt.run_name = run_name
    opt.device = str(device)
    opt.train_method = train_method
    opt.model_name = model.__class__.__name__
    opt.optimizer_name = optimizer.__class__.__name__
    opt.scheduler_name = scheduler.__class__.__name__
    save_dict_as_json(vars(opt), log_dir=log_path, save_name=f'{train_method}_options')

    trainer = ClassificationModelTrainer(model=model, optimizer=optimizer, scheduler=scheduler, dataset='CIFAR10',
                                         batch_size=opt.batch_size, num_workers=opt.num_workers,
                                         data_path=opt.data_path, log_path=log_path, checkpoint_path=checkpoint_path)

    trainer.train_model(num_epochs=opt.num_epochs)
