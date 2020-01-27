"""
Code for training the teacher network on the CIFA10 classification task.
"""
from torch import nn, optim

from utils.init import initialize
from utils.gpu_utils import get_gpu_if_available
from utils.logs import save_dict_as_json
from model_trainers.classify_trainer import ClassificationModelTrainer


def train_classifier(opt, model: nn.Module, train_method: str):
    # Initialize the run.
    run_number, run_name, log_path, checkpoint_path = initialize(opt.record_path, train_method=train_method)
    device = get_gpu_if_available(gpu=opt.gpu)
    print(f'Using device {device}.')

    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)
    scheduler = None  # No learning rate scheduling for now.

    # Saving options and extra information in json file for later viewing.
    opt.run_name = run_name
    opt.device = str(device)
    opt.train_method = train_method
    opt.model_name = model.__class__.__name__
    opt.optimizer_name = optimizer.__class__.__name__
    opt.scheduler_name = scheduler.__class__.__name__
    save_dict_as_json(vars(opt), log_dir=log_path, save_name='classifier_options')

    trainer = ClassificationModelTrainer(model=model, optimizer=optimizer, scheduler=scheduler, dataset='CIFAR10',
                                         batch_size=opt.batch_size, data_path=opt.data_path, log_path=str(log_path),
                                         checkpoint_path=str(checkpoint_path))

    trainer.train_model(num_epochs=opt.num_epochs)
