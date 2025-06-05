from time import time
import cv2
import mediapipe as mp
import pandas as pd
import torch
from constants.constants import CLASSES
from utils.extract_features import extract_features
from utils.numpy_utils import pad_sequence
from utils.pandas_utils import normalize_dataframe


def test_word(model, word, path, count=0, miss=0):
    model.eval()
    buffer = []
    tmp_buffer = []
    print(f"Starting testing {word.upper()}..")
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        cap = cv2.VideoCapture(path)
        # checks whether frames were extracted
        success = 1
        silence = 0
        rise = False
        last_features = [0, 0, 0, 0]
        time1 = time()

        while success:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera image.")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                features, area, _ = extract_features(results, image)
                if not rise:
                    if area > 600:
                        rise = True
                        silence = 0
                        tmp_buffer.append(last_features)

                if rise:
                    tmp_buffer.append(features)
                    if area < 500:
                        silence += 1
                        if silence == 1:
                            if len(tmp_buffer) >= 6:
                                buffer.extend(tmp_buffer)
                                tmp_buffer.clear()
                            else:
                                tmp_buffer.clear()
                                silence = 0
                                rise = False
                        elif silence > 4:
                            rise = False
                            silence = 0
                            normalized = normalize_dataframe(buffer)
                            if 60 > len(buffer) > 15:

                                padded = pad_sequence(pd.DataFrame(normalized), 60)
                                tensor = torch.tensor(padded).float()
                                tensor = tensor.unsqueeze(0)
                                if not (torch.all(torch.eq(tensor, 0))):
                                    with torch.no_grad():
                                        output = model(tensor)

                                    # Get the class with the highest probability
                                    pred = torch.argmax(output).item()
                                    prob = torch.softmax(output, dim=1)[0][pred].item()

                                    # Print the prediction and the confidence
                                    time2 = time() - time1
                                    time1 = time()
                                    print("\nGathered:", str(len(buffer)), "frames in", str(time2), "seconds")
                                    print("Prediction:", CLASSES[pred], "with confidence:", prob, "\n")
                                    if CLASSES[pred] == word:
                                        count += 1
                                    else:
                                        miss += 1
                            buffer.clear()
                            tmp_buffer.clear()
                        else:
                            buffer.append([0, 0, 0, 0])
                            tmp_buffer.clear()
                    else:
                        silence = 0
                last_features = features

                # Exit if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"I was correct {count} times out of {count + miss} times..")