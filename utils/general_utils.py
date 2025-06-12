import os
from typing import List

from constants.constants import CLASSES


def get_encoded_labels():
    encoded_labels= {}
    for i in range(len(CLASSES)):
        encoded_labels[CLASSES[i]] = i
    return encoded_labels


def convert_tuple(tup):
    # initialize an empty string
    str = ''
    for item in tup:
        str = str + item
    return str


def convex_hull(coordinates: List[tuple]) -> List[tuple]:
    def cross_product(a, b, c):
        x1 = b[0] - a[0]
        y1 = b[1] - a[1]
        x2 = c[0] - a[0]
        y2 = c[1] - a[1]
        return x1 * y2 - x2 * y1

    coordinates.sort()
    lower_hull = []
    for p in coordinates:
        while len(lower_hull) >= 2 and cross_product(lower_hull[-2], lower_hull[-1], p) < 0:
            lower_hull.pop()
        lower_hull.append(p)

    upper_hull = []
    for p in reversed(coordinates):
        while len(upper_hull) >= 2 and cross_product(upper_hull[-2], upper_hull[-1], p) < 0:
            upper_hull.pop()
        upper_hull.append(p)

    return lower_hull[:-1] + upper_hull[:-1]

def remove_old_files(full_path):
    for file in os.listdir(full_path):
        file_path = os.path.join(full_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
