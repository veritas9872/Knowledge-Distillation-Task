"""
Code for training the teacher network on the CIFAR10 classification task.
"""
from torchvision.models import vgg16_bn

from utils.options import get_train_options
from train.train_classifier import train_classifier


if __name__ == '__main__':
    train_method = 'Teacher'
    teacher = vgg16_bn(num_classes=10)
    options = get_train_options().parse_args()
    train_classifier(options, model=teacher, train_method=train_method)
