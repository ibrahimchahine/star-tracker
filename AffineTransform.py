from PIL import Image
from math import radians
import numpy as np


def getAffineTransform(src_points, dst_pts):
    src_points = np.array(src_points, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    tx = dst_pts[0][0] - src_points[0][0]
    ty = dst_pts[0][1] - src_points[0][1]
    c = np.cos(radians(dst_pts[1][0] - src_points[1][0]))
    s = np.sin(radians(dst_pts[1][0] - src_points[1][0]))
    A = np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]])
    return A
