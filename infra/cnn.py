import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes, num_rows, num_cols):
        super(CNN, self).__init__()
        hidden_channels = [16, 32]
        hidden_sizes = [(num_rows, num_cols), (num_rows // 2, num_cols // 2)]
        self.fc_input_size = hidden_channels[-1] * hidden_sizes[-1][0] * hidden_sizes[-1][1]
        self.rows = num_rows
        self.cols = num_cols
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(480, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 480)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


