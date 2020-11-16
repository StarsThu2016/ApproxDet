'''''
The detection latency profiler for ApproxDet under contention. 

Usage:
python3 ApproxDetectionProfiler.py --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --weight=models/ApproxDet.pb
'''

from utils_approxdet.contention_generator_3d import contention_generator_launch
from utils_approxdet.contention_generator_3d import contention_generator_kill
import time, subprocess, argparse

# Experiment parameter
WAIT_TIME_TILL_STABLE_SEC = 5

# test_plan3
if 1:
    test_plan_name = "valimg_tp3_hws"
    num_runs, imagefiles = 1, "test/VID_{}.txt".format(test_plan_name)
    gpu_levels = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    test_plan3 = [(0, 0, g) for g in gpu_levels]

# test_plan4
if 0:
    test_plan_name = "testimg_tp4_hws"
    num_runs, imagefiles = 1, "test/VID_{}.txt".format(test_plan_name)
    gpu_levels = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    test_plan4 = [(0, 0, g) for g in gpu_levels]

# Argument parsing
parser = argparse.ArgumentParser(description=('The detection latency profiler '
  'for ApproxDet under contention.'))
parser.add_argument('--dataset_prefix', dest='dataset_prefix',
  help='The path to the dataset.')
parser.add_argument('--weight', dest='weight',
  help='The path to the weight file.')
args = parser.parse_args()

for idx, (cpu_level, memory_bandwidth_level, gpu_level) in enumerate(test_plan3):
    print("Exp #{}, cpu/mem_bw/gpu levels = {}/{}/{}".format(idx, cpu_level,
      memory_bandwidth_level, gpu_level))

    # Launch 3D contention generator
    contention_generator_launch(cpu_level, memory_bandwidth_level, gpu_level)

    # Wait until the 3D contention generator is stable
    time.sleep(WAIT_TIME_TILL_STABLE_SEC)

    # Launch ApproxDetection
    output = "test/VID_{}_c{}_m{}_g{}.txt".format(test_plan_name,
      cpu_level, memory_bandwidth_level, gpu_level)
    cmd = "python3 ApproxDetection.py --imagefiles={} ".format(imagefiles) + \
          "--dataset_prefix={} ".format(args.dataset_prefix) + \
          "--weight={} --repeat={} ".format(args.weight, num_runs) + \
          "--preheat=1 --output={}".format(output)
    p = subprocess.Popen(cmd, shell = True)

    # Wait
    output = p.communicate()[0]

    # Kill 3D contention generator
    contention_generator_kill()
