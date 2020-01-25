import torch
from torchvision.models import vgg16_bn

from utils.checkpoints import load_model_from_checkpoint


def main():
    teacher_checkpoint_path = ''
    # The teacher model type is fixed by the task. While best to set it with options, that is too much work.
    teacher = vgg16_bn(num_classes=10)
    load_model_from_checkpoint(model=teacher, load_dir=teacher_checkpoint_path)



