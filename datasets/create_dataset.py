import os
import torch
from constants.constants import ROOT_PATH, DATA_SAMPLES_PATH
from datasets.dataset import CustomDataset


def create_datasets(train_ratio):
    full_path = os.path.join(ROOT_PATH, 'datasets', )
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    # remove old files
    for file in os.listdir(full_path):
        file_path = os.path.join(full_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Load the train data
    dataset = CustomDataset(root_dir=DATA_SAMPLES_PATH)
    num_samples = len(dataset)
    num_train_samples = int(num_samples * train_ratio)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [num_train_samples, num_samples - num_train_samples])
    torch.save(train_dataset, 'cnn_train_data.pt')
    torch.save(test_dataset, 'cnn_test_data.pt')
    torch.save(dataset, 'cnn_all_data.pt')

    return (train_dataset, test_dataset, dataset), (num_samples, num_train_samples)