import gc
import os
from datetime import datetime
import cv2
from matplotlib import pyplot as plt
from utils.extract_features import mp_face_mesh, extract_features
from utils.pandas_utils import save_features_for_word
from constants.constants import VIDEOS_PATH

class WordDataHandler:
    def __init__(self, word):
        self.mirror_image = None
        self.image = None
        self.word = word
        self.features = []
        self.features1 = []
        self.x_d = []
        self.y_d = []

    def create_data_for_word(self, filename):
        full_raw_path = os.path.join(VIDEOS_PATH, self.word, filename)
        print(f"Creating data for word: {self.word}..")
        print("collecting frames for feature extraction..")
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            vido_object = cv2.VideoCapture(full_raw_path)
            fs = int(vido_object.get(cv2.CAP_PROP_FPS))
            success = 1

            while success:

                success, image = vido_object.read()
                if not success:
                    continue

                image.flags.writeable = True
                self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.mirror_image = cv2.flip(image, flipCode=1)

                results = face_mesh.process(self.image)
                mirror_results = face_mesh.process(self.mirror_image)

                self.add_features(results, mirror_results)

            cv2.destroyAllWindows()

        print(f"Done collecting features from {len(self.x_d)} frames..")
        self.save_features_for_word(word)

    def save_features_for_word(self, word):
        save_features_for_word(word, self.features, self.features1, self.x_d, self.y_d)

    def add_features(self, result, mirror_result):
        if result.multi_face_landmarks:
            features, area, shape = extract_features(result,self.image)
            self.features.append(features)
            self.x_d.append(area)
        if mirror_result is not None:
            features1, area1, shape1 = extract_features(mirror_result, self.mirror_image)
            self.features1.append(features1)
            self.y_d.append(area1)

if __name__ == "__main__":
    for word in os.listdir(VIDEOS_PATH):
        if word == ".DS_Store":
            continue
        now = datetime.now()
        time_formatted = now.strftime("%I:%M %p")
        print("The current time is:", time_formatted)
        for filename in os.listdir(os.path.join(VIDEOS_PATH, word)):
            word_handler = WordDataHandler(word)
            try:
                word_handler.create_data_for_word(filename)
                now = datetime.now()
                time_formatted = now.strftime("%I:%M %p")
                plt.close('all')
                gc.collect()
            except Exception as e:
                print(e)
