// compile: gcc -O3 contention_module_memory_bandwidth.c -o contention_module_memory_bandwidth.o
// run: ./contention_module_memory_bandwidth.o 4092

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h> 

#define N 40000000  // Count of array elements, 40M int --> 160MB
#define min(x,y) (((x)<(y))?(x):(y))
#define max(x,y) (((x)>(y))?(x):(y))

// Return seconds since Epoch.
double mysecond() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char *argv[]) {
    unsigned int i, j;
    double t0, t1, t2;
    double time_diff, bw_temporal;
    static int mem[N];
    int target, permille;

    sscanf(argv[1], "%d", &target);
    printf("Input argument: Target = %d MB/s\n", target);

    //convert the target to permille which between [0, 1024]
    permille = target / 4;
    if (permille > 1024)
        permille = 1024;
    else if (permille < 0)
        permille = 0; 
    t0 = mysecond();
    for (j = 0; 1; j++) {
        t1 = mysecond();
        for (i = 0; i < N; i++) {
            // Write data to array memory.
            if ((i & 1023) < permille) // equivalent to i%1024
                mem[i] = i + j;
        }
        t2 = mysecond();
        time_diff = (t2 - t1);

        bw_temporal = 1.0E-06 * (N >> 10) * max(0,min(permille,1024)) * sizeof(int) / time_diff;
        if (bw_temporal < target - 5){
            if (permille < 1024)
                permille += 1;
        }
        else if (bw_temporal > target + 5){
            if (permille > 0)
                permille -= 1;
        }
    }
    return 0;
}

