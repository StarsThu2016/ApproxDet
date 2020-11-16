# GPU contention generator
# Usage: python3 contention_module_gpu.py --GPU 10

import numpy, math, time, argparse
from numba import cuda

cuda.select_device(0)

# Translation from GPU utiliazation to the internal parameter
GPU_to_num = {
    1: 51200,
    10: 269056,
    20: 532608,
    30: 786816,
    40: 1054464,
    50: 1345792,
    60: 1635840,
    70: 1893632,
    80: 2192128,
    90: 2416768,
    99: 2797696}

# Kernel function
@cuda.jit
def my_kernel(array):

    # CUDA kernel
    pos = cuda.grid(1)
    tx = cuda.threadIdx.x
    if pos < array.size:
        array[pos] += tx  # element add computation

# GPU contention genarator
def GPU_contention_generator(level):

    ContentionSize = GPU_to_num[level]
    data = numpy.zeros(ContentionSize)
    multiplier = data.size / 512    
    threadsperblock, blockspergrid = 128, 4

    # Copy data to device
    device_data = cuda.to_device(data)
    start = time.time()
    while True:
        my_kernel[math.ceil(multiplier*blockspergrid), threadsperblock](device_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=int, help='GPU contention level')
    args = parser.parse_args()
    GPU_contention_generator(args.GPU)

