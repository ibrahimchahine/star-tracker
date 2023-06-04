import numpy as np
from NanoPIAlgorithm import *

algo = Algorithm()
dst_inliner, src_inliners = algo.run_nanopi(
    image1="pics/fr1.jpg", image2="pics/fr2.jpg"
)
