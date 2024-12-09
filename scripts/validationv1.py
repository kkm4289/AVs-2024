"""
This module is to validate if the shift is valid or not
"""

import numpy as np
import cv2


def get_pixel_neighborhood_diff(img1, img2, point1, point2,
                                window_size=5):
    """
    Calculate difference between two points in different images by averaging surrounding pixels.

    Args:
        -img1: First image array
        -img2: Second image array
        -point1: (x,y) coordinates in first image
        -point2: (x,y) coordinates in second image
        -window_size: Size of window around each point (default 5x5)

    """
    x1, y1 = point1  # point1=(x1,y1)
    x2, y2 = point2
    half_window = window_size // 2

    # Extract neighborhoods around each point
    neighborhood1 = img1[max(0, y1 - half_window):min(img1.shape[0], y1 + half_window + 1),
                    max(0, x1 - half_window):min(img1.shape[1], x1 + half_window + 1)]

    neighborhood2 = img2[max(0, y2 - half_window):min(img2.shape[0], y2 + half_window + 1),
                    max(0, x2 - half_window):min(img2.shape[1], x2 + half_window + 1)]

    # Convert to grayscale if images are in color
    if len(img1.shape) == 3:
        neighborhood1 = cv2.cvtColor(neighborhood1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        neighborhood2 = cv2.cvtColor(neighborhood2, cv2.COLOR_BGR2GRAY)

    # AVG
    avg1 = np.mean(neighborhood1)
    avg2 = np.mean(neighborhood2)

    return abs(avg1 - avg2)


def validate_transformation_with_averaging(img1, img2,
                                           points1, points2,
                                           threshold=50.0):
    """
    Validate transformation by comparing neighborhood averages of corresponding points.

    Args:
        img1: First image
        img2: Second image
        points1: Array of points in first image
        points2: Array of points in second image
        threshold: Maximum allowed difference (0-255)

    Returns:
        bool: True if transformation seems valid
    """
    differences = []

    for (p1, p2) in zip(points1, points2):
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        if (0 <= x1 < img1.shape[1] and 0 <= y1 < img1.shape[0] and
                0 <= x2 < img2.shape[1] and 0 <= y2 < img2.shape[0]):
            diff = get_pixel_neighborhood_diff(img1, img2, (x1, y1), (x2, y2))
            differences.append(diff)

    avg_diff = np.mean(differences) if differences else float('inf')

    return avg_diff < threshold, avg_diff
