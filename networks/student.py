from torch import nn, Tensor


class StudentBlock(nn.Module):
    """
    Basic block for the student network.
    Conv, BN, ReLU, MP.
    All conv layers have 3x3 kernels.
    Max poling is 2x2.
    Zero padding included for smoother size reduction.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.layers(tensor)


class StudentNet(nn.Module):
    """
    Student network for knowledge distillation.
    All blocks have 64 channel depth.
    Input size must be 32x32 because of the fully connected layer.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_layers = nn.Sequential(  # 32x32 input
            StudentBlock(in_channels=in_channels, out_channels=64),  # 16x16 output
            StudentBlock(in_channels=64, out_channels=64),  # 8x8 output
            StudentBlock(in_channels=64, out_channels=64),  # 4x4 output
            StudentBlock(in_channels=64, out_channels=64),  # 2x2 output
        )
        # The Fully Connected layer assumes input of size 32x32. 64*4=256.
        self.linear = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.linear(self.conv_layers(tensor).view(tensor.size(0), -1))  # Flatten with .view() method.
