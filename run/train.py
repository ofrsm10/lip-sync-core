import os.path
from time import time
import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
from constants.constants import CLASSES, MODEL_PATH
from model.cnn import CNN
from datasets.create_dataset import create_datasets
from utils.general_utils import get_encoded_labels, convert_tuple
from utils.plots import plot_train_loss, plot_train_accuracy


def train(num_epochs=40,train_ratio=0.7):
    """ Train the CNN model on the dataset."""
    (train_dataset, test_dataset, dataset), (num_samples, num_train_samples) = create_datasets(train_ratio)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    print(f"Starting Training {num_train_samples * num_epochs} samples..\n")
    time1 = time()
    train_loss = []
    train_acc = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for i, (input, clas) in enumerate(train_loader):
            # Encode labels
            label = [get_encoded_labels()[convert_tuple(clas)]]
            label = torch.tensor(label, dtype=torch.long)
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # outputs = [model(inp) for inp in inputs]
            outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += label.size(0)
            epoch_correct += predicted.eq(label).sum().item()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

        epoch_loss /= len(train_loader)
        epoch_acc = 100. * epoch_correct / epoch_total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

    print(f"Finished Training after {time() - time1} seconds ..\n")
    print("\n")
    plot_train_loss(train_loss)
    plot_train_accuracy(train_acc)

    return model

if __name__ == "__main__":
    epochs = int(input("How many epochs? "))
    model = train(num_epochs=epochs)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH,"model.pth"))
    print(f"Model saved as {os.path.join(MODEL_PATH,'model.pth')}")
