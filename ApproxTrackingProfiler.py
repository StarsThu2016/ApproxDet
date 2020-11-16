'''''
The tracking latency profiler for ApproxDet under contention. 

Usage:
python3 ApproxTrackingProfiler.py --dataset_prefix=/home/nvidia/ILSVRC2015/
'''

from utils_approxdet.contention_generator_3d import contention_generator_launch
from utils_approxdet.contention_generator_3d import contention_generator_kill
import time, subprocess, argparse

# Experiment parameter
WAIT_TIME_TILL_STABLE_SEC = 15

# test_plan2
if 1:
    test_plan_name = "valimg_trtp2_hws"
    detection_file = "test/VID_valset_nprop100_shape576_det.txt"
    num_runs, si, imagefiles = 1, 8, "test/VID_{}.txt".format(test_plan_name)
    cpu_levels = [1, 2, 3, 4, 5, 6]
    mem_levels = [600, 1200, 1800, 2400, 3000, 3600]
    gpu_levels = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    test_plan2 = [(0, 0, 0)] + \
                 [(c, m, 0) for c in cpu_levels for m in mem_levels] + \
                 [(0, 0, g) for g in gpu_levels]

# test_plan3
if 0:
    test_plan_name = "testimg_trtp3_hws"
    detection_file = "test/VID_testset_nprop100_shape576_det.txt"
    num_runs, si, imagefiles = 1, 8, "test/VID_{}.txt".format(test_plan_name)
    cpu_levels = [1, 2, 3, 4, 5, 6]
    mem_levels = [600, 1200, 1800, 2400, 3000, 3600]
    gpu_levels = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    test_plan3 = [(0, 0, 0)] + \
                 [(c, m, 0) for c in cpu_levels for m in mem_levels] + \
                 [(0, 0, g) for g in gpu_levels]

# test_plan4
if 0:
    test_plan_name = "valimg_trtp4_hws"
    detection_file = "test/VID_valset_nprop100_shape576_det.txt"
    num_runs, si, imagefiles = 1, 8, "test/VID_{}.txt".format(test_plan_name)
    cpu_levels = [1, 2, 3, 4, 5, 6]
    mem_levels = [600, 1200, 1800, 2400, 3000, 3600, 4000]
    gpu_levels = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    test_plan4 = [(0, 0, 0)] + \
                 [(c, c*m, 0) for c in cpu_levels for m in mem_levels]

# test_plan5
if 0:
    test_plan_name = "valimg_trtp5_hws"
    detection_file = "test/VID_valset_nprop100_shape576_det.txt"
    num_runs, si, imagefiles = 1, 8, "test/VID_{}.txt".format(test_plan_name)
    test_plan5 = [(0, 0, 0), (6, 0, 0), (6, 24000, 0), (0, 0, 50)]

# Argument parsing
parser = argparse.ArgumentParser(description=('The tracking latency profiler '
  'for ApproxDet under contention.'))
parser.add_argument('--dataset_prefix', dest='dataset_prefix',
  help='The path to the dataset.')
args = parser.parse_args()

for idx, (cpu_level, memory_bandwidth_level, gpu_level) in enumerate(test_plan2):
    print("Exp #{}, cpu/mem_bw/gpu levels = {}/{}/{}".format(idx, cpu_level,
      memory_bandwidth_level, gpu_level))

    # Launch 3D contention generator
    contention_generator_launch(cpu_level, memory_bandwidth_level, gpu_level)

    # Wait until the 3D contention generator is stable
    time.sleep(WAIT_TIME_TILL_STABLE_SEC)

    # Launch ApproxTracking
    output = "test/VID_{}_si{}_c{}_m{}_g{}.txt".format(test_plan_name, si,
      cpu_level, memory_bandwidth_level, gpu_level)

    cmd = "python3 ApproxTracking.py --imagefiles={} ".format(imagefiles) + \
          "--detection_file={} ".format(detection_file) + \
          "--dataset_prefix={} ".format(args.dataset_prefix) + \
          "--repeat={} --si={} ".format(num_runs, si) + \
          "--output={}".format(output)
    p = subprocess.Popen(cmd, shell = True)

    # Wait
    output = p.communicate()[0]

    # Kill 3D contention generator
    contention_generator_kill()

