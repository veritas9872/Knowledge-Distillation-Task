import torch
# from ax import optimize


if __name__ == '__main__':
    # Reproducibility settings to reduce variability while searching for best parameters.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed=9872)
