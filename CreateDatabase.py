import numpy as np

from Algorithm import *
import os

dir = "pics"
algo = Algorithm()
filename = "database_class_exmaple.npy"

data = []
for image in os.listdir(dir):
    print("Stars detecting in" + image)
    # Detecting the stars
    stars1 = algo.detect(image=dir + "/" + image)
    temp = algo.stars_list_to_array(stars=stars1)
    data.append(np.array(temp))

# Saving the arrays to the file.
np.save(filename, data, allow_pickle=True)
# Loading the arrays from the file.
database = np.load(filename, allow_pickle=True)
print(database)
