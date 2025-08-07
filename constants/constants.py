"""
Constants module for lip-sync core project.

This module contains all the constant values used throughout the lip-sync project,
including MediaPipe FaceMesh landmarks, class definitions, and file paths.
"""

import os
from typing import FrozenSet, Tuple, List

# MediaPipe FaceMesh lip landmarks for facial feature extraction
FACEMESH_LIPS: FrozenSet[Tuple[int, int]] = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375),
    (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
    (82, 13), (13, 312), (312, 311), (311, 310),
    (310, 415), (415, 308)
])

# Lip landmark indices for feature extraction
INLINE: List[int] = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]
OUTLINE: List[int] = [0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 291, 314, 321, 375, 405, 409]

# Key facial landmarks
UPPER: int = 13
LOWER: int = 14
LEFT: int = 78
RIGHT: int = 308
TOP_RIGHT: int = 80
TOP_LEFT: int = 310
BOTTOM_RIGHT: int = 88
BOTTOM_LEFT: int = 402

# Processing parameters
CUTOFF: int = 1

# Word classes for lip-reading classification (Hebrew words)
CLASSES: List[str] = ["אחד", "שתיים", "חתול", "אבא", "כלב", "פיל", "אריה", "עופר"]

# Project directory paths
ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SAMPLES_PATH: str = os.path.join(ROOT_PATH, "data_samples")
DATA_SETS_PATH: str = os.path.join(ROOT_PATH, "datasets")
MODEL_PATH: str = os.path.join(ROOT_PATH, "cnn_model")
STATS_PATH: str = os.path.join(ROOT_PATH, "stats")
VIDEOS_PATH: str = os.path.join(ROOT_PATH, "videos")
