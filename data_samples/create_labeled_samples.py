import gc
import os
from datetime import datetime
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from Utils.extract_features import mp_face_mesh, extract_features
from Utils.pandas_utils import crop_sequences
from Utils.plots import save_boundary_plot, save_files_and_plots
from constants.constants import VIDEOS_PATH


def create_data_for_word(raw_path, word, filename):
    x_d = []
    y_d = []
    frames_features = []
    frames_features1 = []
    full_raw_path = os.path.join(raw_path, word, filename)
    print(f"Creating data for word: {word}..")
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mirror_image = cv2.flip(image, flipCode=1)

            results1 = face_mesh.process(image)
            if results1.multi_face_landmarks:
                features, area, shape = extract_features(results1, image)
                frames_features.append(features)
                x_d.append(area)

            results2 = face_mesh.process(mirror_image)
            if results2.multi_face_landmarks:
                features1, area1, shape1 = extract_features(results2, mirror_image)
                frames_features1.append(features1)
                y_d.append(area1)

        cv2.destroyAllWindows()

    print(f"Done collecting features from {len(x_d)} frames..")
    word_path = f'/Users/ofersimchovitch/PycharmProjects/lipSyncBeta/Data/{word}'
    if not os.path.exists(word_path):
        os.mkdir(word_path)

    k = 1
    while True:
        new_dir1 = "iteration" + "{:03d}".format(k)
        if not os.path.exists(f'{word_path}/{new_dir1}'):
            os.mkdir(f'{word_path}/{new_dir1}')
            new_dir2 = new_dir1 + "_mirror"
            os.mkdir(f'{word_path}/{new_dir2}')
            break
        k += 1

    with open(f'{word_path}/{new_dir1}/features_raw.csv', 'w', newline='') as file:
        features_df = pd.DataFrame(frames_features)
        features_df.to_csv(file)
    with open(f'{word_path}/{new_dir2}/features_raw.csv', 'w', newline='') as file:
        m_features_df = pd.DataFrame(frames_features1)
        m_features_df.to_csv(file)

    data3 = {"features": frames_features, "area": x_d, "area1": y_d}
    collection_df = pd.DataFrame(data3)
    save_boundary_plot(collection_df, f"{word_path}")

    cropped_features, cropped_m_features = crop_sequences(collection_df, features_df, m_features_df)

    save_files_and_plots(cropped_features, cropped_m_features, f"{word_path}/{new_dir1}")

if __name__ == "__main__":
    for word in os.listdir(VIDEOS_PATH):
        if word == ".DS_Store":
            continue
        now = datetime.now()
        time_formatted = now.strftime("%I:%M %p")
        print("The current time is:", time_formatted)
        for filename in os.listdir(os.path.join(VIDEOS_PATH, word)):
            try:
                create_data_for_word(VIDEOS_PATH, word, filename)
                now = datetime.now()
                time_formatted = now.strftime("%I:%M %p")
                plt.close('all')
                gc.collect()
            except Exception as e:
                print(e)
