from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomCrop


def get_cifar10_loaders(data_root: str, batch_size: int, num_workers: int, augment=True) -> (DataLoader, DataLoader):
    """ Function for retrieving CIFAR10 data as data loaders.

    Training set is augmented. Note that ToTensor() automatically divides by 255.
    Args:
        data_root: Path to CIFAR10 data.
        batch_size: Batch size of data.
        num_workers: Number of workers to pre-process data.
        augment: Whether to use data augmentation on the training data.

    Returns:
        Training and evaluation data loaders for the CIFAR10 dataset.
    """
    # Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    if augment:
        train_transform = Compose([RandomHorizontalFlip(), RandomCrop(size=32, padding=4), ToTensor()])
    else:
        train_transform = ToTensor()
    eval_transform = ToTensor()

    train_set = CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
    eval_set = CIFAR10(root=data_root, train=False, transform=eval_transform, download=True)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, eval_loader
