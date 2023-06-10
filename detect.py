import numpy as np
from NanoPIAlgorithm import *

filename = "database.npy"
image = "pics/fr1.jpg"

algo = Algorithm()
final_src_inliers, final_dst_inliers = algo.run_algo(filename=filename, image=image)
print(final_src_inliers, final_dst_inliers)
