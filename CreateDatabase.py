import numpy as np
from NanoPIAlgorithm import *

algo = Algorithm()
stars1 = algo.detect_nanopi(img=image1)
stars2 = algo.detect_nanopi(img=image2)
print("Stars detected in image1: " + str(len(stars1)) + " image1: " + str(len(stars2)))
