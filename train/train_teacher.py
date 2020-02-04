"""
Code for training the teacher network on the CIFAR10 classification task.
"""
import torch
from torchvision.models import vgg16_bn

from utils.options import classification_options
from train.train_classifier import train_classifier


if __name__ == '__main__':
    train_method = 'Teacher'
    # torch.backends.cudnn.benchmark = True  # Increase speed if input sizes are the same.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.random.manual_seed(9872)
    teacher = vgg16_bn(num_classes=10)  # Settings for the CIFAR10 dataset.
    options = dict(
        num_epochs=400,
        batch_size=512,
        num_workers=4,
        lr=0.1,
        gpu=1
    )
    options = classification_options(**options).parse_args()
    train_classifier(options, model=teacher, train_method=train_method)
