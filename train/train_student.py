"""
Code for training the student network on the CIFAR10 classification task.
The main purpose of this script is as a benchmark to compare knowledge distillation with.
"""
import torch
from networks.student import StudentNet

from utils.options import classification_options
from train.train_classifier import train_classifier


if __name__ == '__main__':
    train_method = 'Student'
    # torch.backends.cudnn.benchmark = True  # Increase speed if input sizes are the same.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.random.manual_seed(9872)
    student = StudentNet(in_channels=3, num_classes=10)  # Settings for the CIFAR10 dataset.
    options = dict()
    options = classification_options(**options).parse_args()
    train_classifier(options, model=student, train_method=train_method)
