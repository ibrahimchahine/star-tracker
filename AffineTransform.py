from PIL import Image
from math import radians, sqrt, pow
import numpy as np


def dist(point1, point2):
    """Calculate the Euclidean distance between two points."""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions.")

    squared_sum = 0
    for i in range(len(point1)):
        squared_sum += pow((point1[i] - point2[i]), 2)

    return sqrt(squared_sum)


def getAffineTransform(points1, points2):
    src_pts = np.float32(points1)
    dst_pts = np.float32(points2)
    A = np.array(
        [
            [src_pts[0][0], src_pts[0][1], 1, 0, 0, 0],
            [0, 0, 0, src_pts[0][0], src_pts[0][1], 1],
            [src_pts[1][0], src_pts[1][1], 1, 0, 0, 0],
            [0, 0, 0, src_pts[1][0], src_pts[1][1], 1],
            [src_pts[2][0], src_pts[2][1], 1, 0, 0, 0],
            [0, 0, 0, src_pts[2][0], src_pts[2][1], 1],
        ]
    )

    B = np.array(
        [
            dst_pts[0][0],
            dst_pts[0][1],
            dst_pts[1][0],
            dst_pts[1][1],
            dst_pts[2][0],
            dst_pts[2][1],
        ]
    )
    M = np.linalg.solve(A, B)
    M = np.reshape(M, (2, 3))

    return M
