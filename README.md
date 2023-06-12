# Star Tracker - By Ibrahim Chahine & Yehonatan Amosi
This repository represents our star tracker implemention in python for the final project in the _Intro To New Space course_.
Below you will find the explanation about the algorithm that we used with the help of our instructor, and examples.

## The NanoPi board
In this project we made the algorithm for the nanopi board, we used the PILLOW library to detect stars, more details about the star detection can be found [here](https://github.com/ibrahimchahine/star-tracker/blob/main/NanoPIAlgorithm.py).

### How to run the camera?
Go to the mjpg-streamer folder using cd mjpg-streamer, then type make clean all, and then you can type ./start.sh. After the camera is running you can access it in your browser by the you ip and port 8080. 

### Libraries on the nanopi board?
If you want to run this project on the nanopi board you need to make sure that you have the libraries. Please download these libraries using this:

   Numpy: check for the version that is supported by python 3.4 [here](https://pypi.org/project/numpy/#history).
    
   Scipy.spatial: use pip install.
    
   PIL: check for the version that is supported by python 3.4 [here](https://pillow.readthedocs.io/en/stable/installation.html).

### Capturing images using the board camera
After you turn on the camera please navigate to the [NanoPi](https://github.com/ibrahimchahine/star-tracker/tree/main/NanoPi) folder, make sure that you have a images folder if not make one. Then access the snapshot code and change the if to your ip
```sh
import os
import time

snapshot_call = "wget http://[Your Ip Address]:8080/?action=snapshot -O"
n = 10

for i in range(n):
    call = os.system(snapshot_call + " images/" + str(i) + "_output.jpg")
    time.sleep(5)

print("Done")
```
This is an example script that captures 10 images.

## Algorithm
    Get two sets of points represented as stars. Perform 1000 iterations with the following steps:

    1. Randomly select a star from the source set.
    
    2. Randomly select a star from the destination set.
    
    3. Find the two nearest neighbors for each star and calculate the corresponding angles (a1 and a2).
    
    4. If the absolute difference between a1 and a2 is less than the angle threshold, proceed with the following: 
        
        a. Construct a transformation matrix using the six points from the images. 
    
        b. Iterate through the points from src to dst and check if the transformed point is close enough to the dst point, using a threshold. 
    
        c. If the transformed point satisfies the threshold condition, add it to the set of inliers.
    
    
    5. inliers If the new set of inliers is larger than the last one, update the best inliers.

    Return the set of inliers as the final result.

## Example using the Database
In this part, you can see an example where we run the algorithm on on image and the database. In the vidoe you can see that every a couple of seconds in the stars that got detected get changed, this happens because the algorithm check on every array in the database for the best result.

[Example run using the database](https://github.com/ibrahimchahine/star-tracker/assets/22155702/b5e1337e-67db-4031-ab32-99e9ea3f53ac)

## Examples with two images
In this part we will list our results of the tests that we made.
If you want to run the yourself, then please see the _Installation and Run_ section bellow, this way you can run you own tests.
Here are some results we got, for more images check the results folder:
Src Stars            |  Dst Stars
:-------------------------:|:-------------------------:
![src](https://github.com/ibrahimchahine/star-tracker/blob/main/results/src.png)  |  ![dst](https://github.com/ibrahimchahine/star-tracker/blob/main/results/dst.png)
![src](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/src2.png)  |  ![dst](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/dst2.png)
![src](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/src3.png)  |  ![dst](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/dst3.png)
![src](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/src9.png)  |  ![dst](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/dst9.png)
![src](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/src6.png)  |  ![dst](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/dst6.png)
![src](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/src7.png)  |  ![dst](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/dst7.png)
![src](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/src8.png)  |  ![dst](https://github.com/ibrahimchahine/star-tracker-ex1/blob/main/results/dst8.png)
## Installation and Run
Our star tracker uses these python packages: numpy, scipy.spatial, PIL.
If you want to run the code then look at the detect.py file for example
```sh
import numpy as np
from NanoPIAlgorithm import *

filename = "database.npy"
image = "pics/fr1.jpg"

algo = Algorithm()
final_src_inliers, final_dst_inliers = algo.run_algo(filename=filename, image=image)
print(final_src_inliers, final_dst_inliers)
```


## Authors
Ibrahim Chahine, Yehonatan Amosi
