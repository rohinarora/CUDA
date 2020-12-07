## Convolution With Shared Memory

### Source files

* ```convBaseline.cu```. Naive conv of 1D vector with identity conv kernel. 
	* Mutiple threads read same elements from global memory. Reuse operations.
* ```convBaselineShared.cu```. Conv with shared memory
	* Load into shared memory, aka programmable cache. Corner cases arising due to shared memory

### Usage

Compile
```
nvcc -O3 -arch=sm_35 -lineinfo convBaseline.cu -o convBaseline
nvcc -O3 -arch=sm_35 -lineinfo convShared.cu -o convShared
```

Run
```
./<executable_name> <vector_size>
cuda-memcheck ./convShared 100000000 #sanity check
./convBaseline 100000000
nvprof ./convBaseline 100000000 #profiling
```

### Results

```Hardware - Tesla K20m, CUDA Version: 11.0```
* 1D conv of vector size 100M elements with 1x5 kernel
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

* Further optimization possible

investigate which metrics improved compared to the naïve implementation using nvprof




### Profiling

Profiling conv with shared memory
```
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