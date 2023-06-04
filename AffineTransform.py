from PIL import Image
from math import radians
import numpy as np


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
