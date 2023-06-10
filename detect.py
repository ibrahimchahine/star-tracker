import numpy as np
from NanoPIAlgorithm import *

filename = "database.npy"
database = np.load(filename, allow_pickle=True)
algo = Algorithm()
max = 0
final_src_inliers = []
final_dst_inliers = []
for stars in database:
    data = [tuple(i) for i in stars]
    if len(data) > 2:
        dst_inliner, src_inliners = algo.run_nanopi_from_array(
            image1="NanoPi/images/MoonPic.jpeg", stars=data
        )
        if len(dst_inliner) > max:
            final_src_inliers = src_inliners
            final_dst_inliers = dst_inliner
print(final_src_inliers, final_dst_inliers)
