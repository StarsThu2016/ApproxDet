'''
3D contention generator:
The optional argument "GPU" means the GPU utilization in percentage and should
  be one from [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99];
The optional argument "Memory" means the Memory Bandwidth and should be an
  interger from 0 to 18000 (MB/s); 
The optional argument "CPU" means the number of CPU cores and should be one from
  [0, 1, 2, 3, 4, 5, 6].

Usage:
python3 contention_generator_3d.py --GPU 10 --Memory 100 --CPU 2
'''

import subprocess, argparse, os, time

def contention_generator_kill():

    cmd = "pkill -f contention_module"
    p = subprocess.Popen(cmd, shell = True)
    output = p.communicate()[0]
    cmd = "pkill -f gaussian"
    p = subprocess.Popen(cmd, shell = True)
    output = p.communicate()[0]
    cmd = "pkill -f func_anomaly_detection"
    p = subprocess.Popen(cmd, shell = True)
    output = p.communicate()[0]
    time.sleep(1)
    print("All contentions are killed!")
    
def contention_generator_launch(cpu_level, memory_bandwidth_level, gpu_level):

    contention_generator_kill()

    cpu_core_list = [1, 2, 0, 3, 4, 5]
    core_occupied = 0
    default_mb_input_for_cpu_module = 1
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # launch GPU module 
    if gpu_level > 0:
        bin_path = os.path.join(cur_dir, "contention_module_gpu.py")
        core = cpu_core_list[0]
        cmd = "taskset -c {} python3 {} --GPU {}".format(core, bin_path, gpu_level)
        _ = subprocess.Popen(cmd, shell = True)
        core_occupied += 1

    # launch CPU and memory bandwidth module
    cpu_to_split = 1 if cpu_level <= 1 else (cpu_level - core_occupied)
    memory_bandwidth_level_per_cpu = int(memory_bandwidth_level / cpu_to_split)
    bin_path = os.path.join(cur_dir, "contention_module_memory_bandwidth.o")
    if cpu_level <= 1:
        if memory_bandwidth_level > 0:
            cmd = "taskset -c {} {} {}".format(cpu_core_list[0],
                                               bin_path,
                                               memory_bandwidth_level_per_cpu)
            _ = subprocess.Popen(cmd, shell = True)
        else:
            for idx in range(core_occupied, cpu_level):
                cmd = "taskset -c {} {} {}".format(cpu_core_list[idx],
                                                   bin_path,
                                                   default_mb_input_for_cpu_module)
                _ = subprocess.Popen(cmd, shell = True)
    else:
        for idx in range(core_occupied, cpu_level):
            if memory_bandwidth_level > 0:
                cmd = "taskset -c {} {} {}".format(cpu_core_list[idx],
                                                   bin_path,
                                                   memory_bandwidth_level_per_cpu)
            else:
                cmd = "taskset -c {} {} {}".format(cpu_core_list[idx],
                                                   bin_path,
                                                   default_mb_input_for_cpu_module)
            _ = subprocess.Popen(cmd, shell = True)
    print("Contention created: CPU, Memory, GPU = {}, {}, {}".format(cpu_level,
      memory_bandwidth_level, gpu_level))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--Memory', type = int, default = 0,
                        help = 'Memory contention level')
    parser.add_argument('--GPU', type = int, default = 0,
                        help = 'GPU contention level')
    parser.add_argument('--CPU', type = int, default = 0,
                        help='CPU contention level')
    args = parser.parse_args()

    contention_generator_launch(args.CPU, args.Memory, args.GPU)

