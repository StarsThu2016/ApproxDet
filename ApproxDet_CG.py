# The synthetic contention generator
# Usage: python3 ApproxDet_CG.py

from utils_approxdet.contention_generator_3d import contention_generator_launch
from utils_approxdet.contention_generator_3d import contention_generator_kill
import time, os, subprocess

last_line = ""
while True:
    if os.path.isfile("contention_level.txt"):
        with open("contention_level.txt") as f:
            lines = f.readlines()
    else:
        lines = [""]
    if lines[0] != last_line:
        contention_generator_kill()
        items = lines[0].strip().split()
        if len(items) == 1 and items[0] == 'anomaly':
            bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "utils_approxdet/annotation/func_anomaly_detection.py")
            cmd = "python3 " + bin_path
            _ = subprocess.Popen(cmd, shell=True)
        elif len(items) == 1 and items[0] == 'gaussian':
            cmd = "./utils_approxdet/rodinia/gaussian/gaussian -s 8192"
            _ = subprocess.Popen(cmd, shell=True)
        elif len(items) == 3:
            cl, ml, gl = int(items[0]), int(items[1]), int(items[2])
            contention_generator_launch(cl, ml, gl)
        last_line = lines[0]
    time.sleep(2)
