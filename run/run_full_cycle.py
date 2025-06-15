import os
from datetime import datetime
import torch

from cnn_model.cnn import CNN
from constants.constants import CLASSES, MODEL_PATH, VIDEOS_PATH
from evaluate_model import evaluate
from test_real_time import test_word
from train import train

if __name__ == "__main__":
    # epochs = int(input("How many ephocs?"))
    # model = train(num_epochs=epochs, train_ratio=0.7)
    model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "cnn_model.pth")))
    # torch.save(model.state_dict(), os.path.join(MODEL_PATH,"cnn_model.pth"))
    # evaluate(model)
    if input("offline inferece?") == "y":
        raw_path = VIDEOS_PATH
        for word in CLASSES:
            now = datetime.now()
            time_formatted = now.strftime("%I:%M %p")
            print("The current time is:", time_formatted)
            for filename in os.listdir(os.path.join(raw_path, word)):
                full_raw_path = os.path.join(raw_path, word, filename)
                test_word(model=model, word=word, path=full_raw_path)
                now = datetime.now()
                time_formatted = now.strftime("%I:%M %p")
                print("The current time is:", time_formatted)