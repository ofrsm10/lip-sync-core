import cv2
import mediapipe as mp
import numpy as np
from constants.constants import RIGHT, LEFT, UPPER, LOWER, INLINE
from utils.general_utils import convex_hull

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def extract_features(results, frame, teeth_lower=np.array([110, 0, 65]), teeth_upper=np.array([177, 127, 204]),
                     tongue_lower=np.array([0, 20, 60]), tongue_upper=np.array([20, 180, 255])):
    for face_landmarks in results.multi_face_landmarks:
        slopes, idxs = mp_drawing.get_3D_contour_features(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_LIPS)
        x_diff = abs(idxs[RIGHT][0] - idxs[LEFT][0])
        y_diff = abs(idxs[UPPER][1] - idxs[LOWER][1])
        ratio = round(y_diff / x_diff, 4)
        area = x_diff * y_diff
        features = [0, 0, 0, 0]

        if area >= 500:
            if y_diff > 3:
                teeth, tongue, area = extract_mouth_histogram(frame, idxs, tongue_lower, tongue_upper, teeth_lower,
                                                              teeth_upper)
            else:
                teeth = tongue = 0
            features = [ratio, area, teeth, tongue]
        else:
            ratio = area = 0
        return features, area, ratio
    return None


def extract_mouth_histogram(frame, idxs, tongue_lower, tongue_upper,
                            teeth_lower, teeth_upper, index_list=INLINE):
    xi = idxs[LEFT][0]
    xf = idxs[RIGHT][0]
    yi = idxs[UPPER][1]
    yf = idxs[LOWER][1]

    height = frame.shape[0]
    width = frame.shape[1]
    points = []
    for idx in index_list:
        points.append(idxs[idx])
    mask = np.zeros((height, width), dtype=np.uint8)
    hull_points = convex_hull(points)
    points = np.array([hull_points], dtype=int)
    cv2.fillPoly(mask, points, 255)
    total_area = float(np.count_nonzero(mask))

    res = cv2.bitwise_and(frame, frame, mask=mask)
    hsv = cv2.cvtColor(res, cv2.COLOR_RGB2HSV)
    cropped = hsv[yi: yf, xi: xf]

    # Create masks for tongue and teeth regions
    tongue_mask = cv2.inRange(cropped, tongue_lower, tongue_upper)
    teeth_mask = cv2.inRange(cropped, teeth_lower, teeth_upper)

    tongue_area = cv2.countNonZero(tongue_mask)
    teeth_area = cv2.countNonZero(teeth_mask)

    tongue_ratio = round((tongue_area / total_area), 4)
    teeth_ratio = round((teeth_area / total_area), 4)

    return teeth_ratio, tongue_ratio, total_area

