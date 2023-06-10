import numpy as np

from Algorithm import *
import os

dir = "pics"
algo = Algorithm()
filename = "database.npy"

data = []
for image in os.listdir(dir):
    print("Stars detecting in" + image)
    stars1 = algo.detect(image=dir + "/" + image)
    temp = algo.stars_list_to_array(stars=stars1)
    data.append(np.array(temp))

np.save(filename, data, allow_pickle=True)

database = np.load(filename, allow_pickle=True)
print(database)
