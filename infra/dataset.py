"""
Custom dataset class for lip-reading data.

This module provides a PyTorch Dataset implementation for loading
and preprocessing lip-reading features from CSV files.
"""

import os
from typing import Tuple, List, Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.numpy_utils import pad_sequence


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for lip-reading features.
    
    This dataset loads normalized lip features from CSV files and prepares
    them for training the CNN model. Features are padded to a fixed length
    for consistent batch processing.
    
    Args:
        root_dir (str): Root directory containing class subdirectories
    """
    
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.data: List[torch.Tensor] = []
        self.labels: List[str] = []
        
        print(f"Loading dataset from: {root_dir}")
        self._load_data()
        print(f"Dataset loaded: {len(self.data)} samples, {len(set(self.labels))} classes")
    
    def _load_data(self) -> None:
        """Load all feature files from the dataset directory."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        for word in os.listdir(self.root_dir):
            word_path = os.path.join(self.root_dir, word)
            if not os.path.isdir(word_path):
                continue
                
            for subdir, dirs, files in os.walk(word_path):
                for file in files:
                    if file == "features_norm.csv":
                        file_path = os.path.join(subdir, file)
                        try:
                            self._process_feature_file(file_path, word)
                        except Exception as err:
                            print(f"Error processing {file_path}: {err}")
    
    def _process_feature_file(self, file_path: str, word: str) -> None:
        """
        Process a single feature file and add to dataset.
        
        Args:
            file_path (str): Path to the CSV feature file
            word (str): The word label for this sample
        """
        data = pd.read_csv(file_path, header=None, skiprows=1).values[:, 1:]
        padded = pad_sequence(data, 60)
        
        if padded is not None:
            self.data.append(padded)
            self.labels.append(word)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, str]: Feature tensor and label
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
            
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = self.labels[idx]
        return data, label
