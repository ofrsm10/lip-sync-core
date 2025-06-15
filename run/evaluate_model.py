import os
from time import time
import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from cnn_model.cnn import CNN
from constants.constants import CLASSES, DATA_SETS_PATH, MODEL_PATH
from utils.general_utils import get_encoded_labels, convert_tuple
from utils.plots import plot_confusion_matrix, plot_umap


def evaluate(model):
    if not os.path.exists(DATA_SETS_PATH):
        os.mkdir(DATA_SETS_PATH)
    test_dataset = torch.load(os.path.join(DATA_SETS_PATH,"test_data.pt"))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    num_samples = len(test_dataset)
    print(f"Starting Testing {num_samples} samples..\n")
    print("\n")
    time1 = time()
    model.eval()
    test_loss = 0
    correct = 0
    loss_values = []
    accuracy_values = []

    accuracy = 0.0
    with torch.no_grad():
        for test_input, target in test_loader:
            label = [get_encoded_labels()[convert_tuple(target)]]
            label = torch.tensor(label, dtype=torch.long)
            output = model(test_input)
            test_loss += criterion(output, label)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        loss_values.append(test_loss)
        accuracy_values.append(accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            label = [get_encoded_labels()[convert_tuple(label)]]
            label = torch.tensor(label, dtype=torch.long)
            output = model(images)
            _, prediction = torch.max(output, 1)
            # collect the correct predictions for each class
            if label == prediction:
                correct_pred[CLASSES[label]] += 1
            total_pred[CLASSES[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        try:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        except Exception as e:
            print(e)
    features = []
    labels = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            features.append(images.view(images.shape[0], -1).numpy())
            label = [get_encoded_labels()[convert_tuple(targets)]]
            label = torch.tensor(label, dtype=torch.long)
            labels.append(label)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(label.numpy())
            y_pred.extend(predicted.numpy())

    plot_umap(features, labels)
    plot_confusion_matrix(y_true, y_pred)

    print(f"\nFinished Testing after {time() - time1} seconds ..\n")


if __name__ == '__main__':
    test_data_set_path = os.path.join(DATA_SETS_PATH, "test_data.pt")
    if not os.path.exists(test_data_set_path):
        print("No test data found, please run the training script first.")
    else:
        model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "cnn_model.pth")))
        evaluate(model)
        print("Evaluation completed successfully.")
