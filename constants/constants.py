import os

FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])
INLINE = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]
OUTLINE = [0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 291, 314, 321, 375, 405, 409]
UPPER = 13
LOWER = 14
LEFT = 78
RIGHT = 308
TOP_RIGHT = 80
TOP_LEFT = 310
BOTTOM_RIGHT = 88
BOTTOM_LEFT = 402
CUTOFF = 1
CLASSES = ["אחד", "שתיים", "חתול", "אבא", "כלב", "פיל", "אריה", "עופר"]
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_SAMPLES_PATH = os.path.join(ROOT_PATH, "data_samples")
DATA_SETS_PATH = os.path.join(ROOT_PATH, "data_sets")
MODEL_PATH = os.path.join(ROOT_PATH, "model")
STATS_PATH = os.path.join(ROOT_PATH, "stats")
VIDEOS_PATH = os.path.join(ROOT_PATH, "videos")
