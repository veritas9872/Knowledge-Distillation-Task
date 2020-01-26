"""
Code for training the student network on the CIFAR10 classification task.
The main purpose of this script is as a benchmark to compare knowledge distillation with.
"""
from networks.student import StudentNet

from utils.options import classification_options
from train.train_classifier import train_classifier


if __name__ == '__main__':
    train_method = 'Student'
    student = StudentNet(in_channels=3, num_classes=10)
    options = classification_options().parse_args()
    train_classifier(options, model=student, train_method=train_method)
