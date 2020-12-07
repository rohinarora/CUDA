# Convolution With Shared Memory

### Source files

* ```convBaseline.cu```. Standard conv with identity conv kernel. 
	* Mutiple threads read same elements from global memory. Reuse operations.
* ```convBaselineShared.cu```. Conv with shared memory
	* Load into shared memory, aka programmable cache

### Usage

Compile
```
nvcc -O3 -arch=sm_35 -lineinfo convBaseline.cu -o convBaseline
```

Run
```
./<executable file name> <vector size>
./convBaseline 100000000
nvprof ./convBaseline 100000000
nvprof ./convBaselineShared 100000000
```

### Results

* Baseline conv 20x faster on GPU compared to CPU
* Conv with shared mempry 60x faster on GPU compared to CPU
* Further optimization possible

```
(base) [arora.roh@c2138 HOL4]$ ./stencil 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1923.53 ms
Running parallel job.
	Parallel Job Time: 50.76 ms
Correct result. No errors were found.
```


```
(base) [arora.roh@c2138 HOL4]$ ./stencil_shared 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1848.85 ms
Running parallel job.
	Parallel Job Time: 14.66 ms
Correct result. No errors were found.
```







