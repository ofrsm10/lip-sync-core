"""
Convolutional Neural Network model for lip-reading classification.

This module defines a CNN architecture for classifying lip movements
from extracted facial features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNN(nn.Module):
    """
    Convolutional Neural Network for lip-reading classification.
    
    This CNN model processes sequences of lip features to classify spoken words.
    The architecture consists of two convolutional layers followed by max pooling
    and two fully connected layers.
    
    Args:
        num_classes (int): Number of output classes for classification
        num_rows (int): Number of time steps in the input sequence
        num_cols (int): Number of features per time step
    """
    
    def __init__(self, num_classes: int, num_rows: int, num_cols: int) -> None:
        super(CNN, self).__init__()
        
        # Architecture parameters
        hidden_channels = [16, 32]
        hidden_sizes = [(num_rows, num_cols), (num_rows // 2, num_cols // 2)]
        self.fc_input_size = hidden_channels[-1] * hidden_sizes[-1][0] * hidden_sizes[-1][1]
        self.rows = num_rows
        self.cols = num_cols
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(480, 128)  # Fixed size based on expected input dimensions
        self.fc2 = nn.Linear(128, num_classes)
        
        print(f"CNN model initialized with {num_classes} classes, input size: {num_rows}x{num_cols}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, num_rows, num_cols)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Second convolutional block  
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 480)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


