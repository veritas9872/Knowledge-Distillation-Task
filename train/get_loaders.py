from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def get_cifar10_loaders(data_root: str, batch_size: int, num_workers: int) -> (DataLoader, DataLoader):
    # No data augmentation for training data. ToTensor() automatically divides by 255.
    train_set = CIFAR10(root=data_root, train=True, transform=ToTensor(), download=True)
    test_set = CIFAR10(root=data_root, train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    eval_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, eval_loader
