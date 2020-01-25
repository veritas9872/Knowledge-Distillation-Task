from abc import ABC
from typing import Union

import torch
from torch import optim


class ModelTrainer(ABC):
    SchedulerType = Union[optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau]

    def __init__(self):
        torch.backends.cudnn.benchmark = True  # Increases speed assuming input sizes are the same.

    def train_epoch(self):
        pass

    def eval_epoch(self):
        pass

    def train_model(self):
        pass
