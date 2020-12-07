## Convolution With Shared Memory

### Source files

* ```convBaseline.cu```. Naive conv of 1D vector with identity conv kernel. 
	* Mutiple threads read same elements from global memory/Repeated fetch from global memory of same elements.
* ```convBaselineShared.cu```. Conv with shared memory
	* Use shared memory, aka programmable cache

### Usage

Compile
```
nvcc -O3 -arch=sm_35 -lineinfo convBaseline.cu -o convBaseline
nvcc -O3 -arch=sm_35 -lineinfo convShared.cu -o convShared
```

Run
```
./<executable_name> <vector_size>
./convBaseline 100000000
cuda-memcheck ./convShared 100000000 #sanity check
nvprof ./convBaseline 100000000 #profiling
```

### Results

```Hardware - Tesla K20m, CUDA Version: 11.0```
* 1D conv of vector size 100M elements with 1x17 kernel. Below GPU time is kernel time. There is addtional DtoH and HtoD movement overhead reported in profiling below. 
* Baseline conv 30x faster on GPU compared to CPU
```
./convBaseline 100000000 
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1564.89 ms
Running parallel job.
	Parallel Job Time: 51.15 ms
Correct result. No errors were found.
```
* Conv with shared mempry 60x faster on GPU compared to CPU
```
cuda-memcheck ./convShared 100000000
./convShared 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1615.19 ms
Running parallel job.
	Parallel Job Time: 14.89 ms
Correct result. No errors were found.
```

* Both implementations are I/O bound. >90% time is spent in data movement. Breaking the input into CUDA streams is a first possible optimization. 
* The time spent in CUDA kernel call increases as conv kernel size increases (param radius in table below). The advantage of use of shared memory is more evidient in such case
* Using shared memory, less registers per CUDA thread are used (16) than in naive implementation (30)
* Varying the kernel size (kernel is 2*Radius+1). Vector size and Block Size fixed at 100M and 512 resp. Table below

| Radius Size | CPU (ms)    | Naïve GPU (ms) | Optimised GPU (ms)
| ----------- | ----------- |----------- | -----------
| 0      | 200.47       | 11.17 |   11.27
| 1   |     606.68    |  14.23  | 11.81
| 2   |     822.28    |  16.26  | 12.25
| 4   |     1272.61    |  27.28  | 12.60
| 8   |     2155.35    |  51.21  | 14.81
| 16   |     4467.58    |  99.08  | 21.29
| 32   |     8949.27    |  195.68  | 32.92

* As the compute load increases (conv kernel radius), shared memory implementation scales much better than naive version 

### Profiling

**Profiling naive conv**

```
nvprof  ./convBaseline 100000000
==16678== NVPROF is profiling process 16678, command: ./convBaseline 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1615.57 ms
Running parallel job.
	Parallel Job Time: 51.17 ms
Correct result. No errors were found.
==16678== Profiling application: ./convBaseline 100000000
==16678== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   45.27%  240.78ms         1  240.78ms  240.78ms  240.78ms  [CUDA memcpy DtoH]
                   45.20%  240.40ms         1  240.40ms  240.40ms  240.40ms  [CUDA memcpy HtoD]
                    9.52%  50.643ms         1  50.643ms  50.643ms  50.643ms  convBaseline(double*, double*, int)
      API calls:   61.44%  482.10ms         2  241.05ms  240.81ms  241.29ms  cudaMemcpy
                   20.95%  164.36ms         2  82.179ms  1.3350us  164.36ms  cudaEventCreate
                   10.58%  82.981ms         2  41.491ms  1.3815ms  81.600ms  cudaFree
                    6.51%  51.073ms         2  25.536ms  27.093us  51.046ms  cudaEventSynchronize
                    0.44%  3.4532ms         2  1.7266ms  1.5486ms  1.9047ms  cudaMalloc
                    0.03%  258.63us         1  258.63us  258.63us  258.63us  cuDeviceTotalMem
                    0.02%  186.79us       101  1.8490us     227ns  74.530us  cuDeviceGetAttribute
                    0.01%  117.02us         1  117.02us  117.02us  117.02us  cudaLaunchKernel
                    0.01%  75.071us         4  18.767us  7.0160us  31.734us  cudaEventRecord
                    0.00%  27.009us         1  27.009us  27.009us  27.009us  cuDeviceGetName
                    0.00%  12.595us         1  12.595us  12.595us  12.595us  cudaSetDevice
                    0.00%  11.183us         1  11.183us  11.183us  11.183us  cuDeviceGetPCIBusId
                    0.00%  10.159us         2  5.0790us  2.8850us  7.2740us  cudaEventElapsedTime
                    0.00%  2.3440us         3     781ns     313ns  1.2830us  cuDeviceGetCount
                    0.00%  1.2570us         2     628ns     315ns     942ns  cuDeviceGet
                    0.00%     570ns         1     570ns     570ns     570ns  cuDeviceGetUuid
```

Using trace
```
nvprof --print-gpu-trace ./convBaseline 100000000
==16587== NVPROF is profiling process 16587, command: ./convBaseline 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1615.28 ms
Running parallel job.
	Parallel Job Time: 51.14 ms
Correct result. No errors were found.
==16587== Profiling application: ./convBaseline 100000000
==16587== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
6.78879s  239.76ms                    -               -         -         -         -  762.94MB  3.1075GB/s    Pageable      Device   Tesla K20m (0)         1         7  [CUDA memcpy HtoD]
8.64493s  50.645ms         (195313 1 1)       (512 1 1)        30        0B        0B         -           -           -           -   Tesla K20m (0)         1         7  convBaseline(double*, double*, int) [122]
8.69575s  241.54ms                    -               -         -         -         -  762.94MB  3.0847GB/s      Device    Pageable   Tesla K20m (0)         1         7  [CUDA memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy
```

**Profiling conv with shared memory**
```
nvprof ./convShared 100000000
==15132== Profiling application: ./convShared 100000000
==15132== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   48.73%  240.89ms         1  240.89ms  240.89ms  240.89ms  [CUDA memcpy DtoH]
                   48.37%  239.09ms         1  239.09ms  239.09ms  239.09ms  [CUDA memcpy HtoD]
                    2.90%  14.330ms         1  14.330ms  14.330ms  14.330ms  convShared(double*, double*, int)
      API calls:   64.40%  480.76ms         2  240.38ms  239.36ms  241.40ms  cudaMemcpy
                   21.93%  163.74ms         2  81.872ms  1.1820us  163.74ms  cudaEventCreate
                   11.12%  82.986ms         2  41.493ms  1.3729ms  81.614ms  cudaFree
                    1.96%  14.639ms         2  7.3195ms  24.082us  14.615ms  cudaEventSynchronize
                    0.47%  3.4737ms         2  1.7368ms  1.5405ms  1.9332ms  cudaMalloc
                    0.05%  340.83us         1  340.83us  340.83us  340.83us  cudaLaunchKernel
                    0.03%  222.19us         1  222.19us  222.19us  222.19us  cuDeviceTotalMem
                    0.02%  167.60us       101  1.6590us     198ns  67.101us  cuDeviceGetAttribute
                    0.01%  100.33us         4  25.081us  9.2600us  31.479us  cudaEventRecord
                    0.00%  25.433us         1  25.433us  25.433us  25.433us  cudaSetDevice
                    0.00%  23.282us         1  23.282us  23.282us  23.282us  cuDeviceGetName
                    0.00%  10.370us         1  10.370us  10.370us  10.370us  cuDeviceGetPCIBusId
                    0.00%  10.281us         2  5.1400us  2.5610us  7.7200us  cudaEventElapsedTime
                    0.00%  9.9900us         1  9.9900us  9.9900us  9.9900us  cudaDeviceSetSharedMemConfig
                    0.00%  2.0460us         3     682ns     309ns  1.0900us  cuDeviceGetCount
                    0.00%  1.0690us         2     534ns     280ns     789ns  cuDeviceGet
                    0.00%     406ns         1     406ns     406ns     406ns  cuDeviceGetUuid
```

Using trace
```
nvprof --print-gpu-trace ./convShared 100000000
==15687== NVPROF is profiling process 15687, command: ./convShared 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1568.64 ms
Running parallel job.
==15687== Warning: Profiling results might be incorrect with current version of nvcc compiler used to compile cuda app. Compile with nvcc compiler 9.0 or later version to get correct profiling results. Ignore this warning if code is already compiled with the recommended nvcc version 
	Parallel Job Time: 15.26 ms
Correct result. No errors were found.
==15687== Profiling application: ./convShared 100000000
==15687== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
6.90051s  239.26ms                    -               -         -         -         -  762.94MB  3.1140GB/s    Pageable      Device   Tesla K20m (0)         1         7  [CUDA memcpy HtoD]
8.70987s  14.325ms         (195313 1 1)       (512 1 1)        16  4.1250KB        0B         -           -           -           -   Tesla K20m (0)         1         7  convShared(double*, double*, int) [123]
8.72436s  240.97ms                    -               -         -         -         -  762.94MB  3.0919GB/s      Device    Pageable   Tesla K20m (0)         1         7  [CUDA memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy
```

* Both implmentations have almost same CUDA memcpy DtoH and CUDA memcpy HtoD (as expected)

#### Ref

 * Idiom :  Aggregate all constants – values, constant pointers, etc. – into a single constant structure and pass that constant “environment” down through the kernel’s device functions as needed (while being careful to maintain its const’ness). The compiler does a great job recognizing that these values and pointers reside in the constant memory space. Works well for sm_20+. https://forums.developer.nvidia.com/t/defining-global-variables-on-the-host-and-device-at-once/31409/2?u=rohinarora07