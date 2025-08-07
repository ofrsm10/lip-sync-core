"""
Feature extraction module for lip-reading analysis.

This module provides functions for extracting lip movement features from video frames
using MediaPipe face landmarks and OpenCV image processing techniques.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Any
from constants.constants import RIGHT, LEFT, UPPER, LOWER, INLINE
from utils.general_utils import convex_hull

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def extract_features(results: Any, frame: np.ndarray, 
                    teeth_lower: np.ndarray = np.array([110, 0, 65]), 
                    teeth_upper: np.ndarray = np.array([177, 127, 204]),
                    tongue_lower: np.ndarray = np.array([0, 20, 60]), 
                    tongue_upper: np.ndarray = np.array([20, 180, 255])) -> Optional[Tuple[List[float], float, float]]:
    """
    Extract lip movement features from a video frame.
    
    This function analyzes facial landmarks detected by MediaPipe to extract
    four key features: lip ratio, mouth area, teeth visibility, and tongue visibility.
    
    Args:
        results (Any): MediaPipe face detection results
        frame (np.ndarray): Input video frame
        teeth_lower (np.ndarray): Lower HSV threshold for teeth detection
        teeth_upper (np.ndarray): Upper HSV threshold for teeth detection
        tongue_lower (np.ndarray): Lower HSV threshold for tongue detection
        tongue_upper (np.ndarray): Upper HSV threshold for tongue detection
        
    Returns:
        Optional[Tuple[List[float], float, float]]: Tuple containing:
            - features: [ratio, area, teeth, tongue] or None if no face detected
            - area: Mouth bounding box area
            - ratio: Vertical to horizontal lip ratio
    """
    if not results.multi_face_landmarks:
        return None
        
    for face_landmarks in results.multi_face_landmarks:
        # Get 3D contour features from MediaPipe
        slopes, idxs = mp_drawing.get_3D_contour_features(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_LIPS
        )
        
        # Calculate lip dimensions
        x_diff = abs(idxs[RIGHT][0] - idxs[LEFT][0])
        y_diff = abs(idxs[UPPER][1] - idxs[LOWER][1])
        
        # Avoid division by zero
        if x_diff == 0:
            ratio = 0.0
        else:
            ratio = round(y_diff / x_diff, 4)
            
        area = x_diff * y_diff
        features = [0.0, 0.0, 0.0, 0.0]

        # Extract detailed features only for significant mouth openings
        if area >= 500:
            if y_diff > 3:
                teeth, tongue, area = extract_mouth_histogram(
                    frame, idxs, tongue_lower, tongue_upper, teeth_lower, teeth_upper
                )
            else:
                teeth = tongue = 0.0
            features = [ratio, area, teeth, tongue]
        else:
            ratio = area = 0.0
            
        print(f"Extracted features: ratio={ratio:.4f}, area={area:.1f}, teeth={features[2]:.4f}, tongue={features[3]:.4f}")
        return features, area, ratio
    
    return None


def extract_mouth_histogram(frame: np.ndarray, idxs: dict, 
                           tongue_lower: np.ndarray, tongue_upper: np.ndarray,
                           teeth_lower: np.ndarray, teeth_upper: np.ndarray, 
                           index_list: List[int] = INLINE) -> Tuple[float, float, float]:
    """
    Extract histogram-based features from the mouth region.
    
    Analyzes the mouth region to detect teeth and tongue visibility
    using HSV color space thresholding.
    
    Args:
        frame (np.ndarray): Input video frame
        idxs (dict): Dictionary of facial landmark indices and coordinates
        tongue_lower (np.ndarray): Lower HSV threshold for tongue detection
        tongue_upper (np.ndarray): Upper HSV threshold for tongue detection
        teeth_lower (np.ndarray): Lower HSV threshold for teeth detection
        teeth_upper (np.ndarray): Upper HSV threshold for teeth detection
        index_list (List[int]): List of landmark indices defining mouth region
        
    Returns:
        Tuple[float, float, float]: (teeth_ratio, tongue_ratio, mouth_area)
    """
    # Get mouth bounding box coordinates
    xi = idxs[LEFT][0]
    xf = idxs[RIGHT][0]
    yi = idxs[UPPER][1]
    yf = idxs[LOWER][1]

    height, width = frame.shape[:2]
    
    # Create mouth region mask
    points = [idxs[idx] for idx in index_list]
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

