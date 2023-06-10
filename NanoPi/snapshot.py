import os
import time


snapshot_call = "wget http://192.168.51.149:8080/?action=snapshot -O"

n = 10

for i in range(n):
    call = os.system(snapshot_call + " images/" + str(i) + "_output.jpg")
    time.sleep(5)

print("Done")
