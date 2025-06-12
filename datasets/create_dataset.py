import os
import torch
from torch.utils.data import random_split
from constants.constants import ROOT_PATH, DATA_SAMPLES_PATH
from datasets.dataset import CustomDataset
from utils.general_utils import remove_old_files


def create_datasets(train_ratio):
    full_path = os.path.join(ROOT_PATH, 'datasets', )
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    remove_old_files(full_path)

    # Load the train data
    dataset = CustomDataset(root_dir=DATA_SAMPLES_PATH)
    num_samples = len(dataset)
    num_train_samples = int(num_samples * train_ratio)
    train_dataset, test_dataset = random_split(dataset,[num_train_samples, num_samples - num_train_samples])
    torch.save(train_dataset, 'cnn_train_data.pt')
    torch.save(test_dataset, 'cnn_test_data.pt')
    torch.save(dataset, 'cnn_all_data.pt')

    return (train_dataset, test_dataset, dataset), (num_samples, num_train_samples)
