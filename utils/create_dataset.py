import os
import torch
from torch.utils.data import random_split
from constants.constants import ROOT_PATH, DATA_SAMPLES_PATH, DATA_SETS_PATH
from utils.dataset import CustomDataset
from utils.general_utils import remove_old_files


def create_datasets(train_ratio):
    dataset = CustomDataset(root_dir=DATA_SAMPLES_PATH)
    num_samples = len(dataset)
    num_train_samples = int(num_samples * train_ratio)
    train_dataset, test_dataset = random_split(dataset,[num_train_samples, num_samples - num_train_samples])
    torch.save(train_dataset, os.path.join(DATA_SETS_PATH,'train_data.pt'))
    torch.save(test_dataset, os.path.join(DATA_SETS_PATH,'test_data.pt'))
    torch.save(dataset, os.path.join(DATA_SETS_PATH,'all_data.pt'))

    return (train_dataset, test_dataset, dataset), (num_samples, num_train_samples)

if __name__ == '__main__':
    remove_old_files(DATA_SETS_PATH)
    train_ratio = float(input("Enter the training ratio (e.g., 0.7 for 70% training data): "))
    create_datasets(train_ratio)
    print("Datasets created successfully.")
