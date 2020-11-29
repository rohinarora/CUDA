### Convolution Speed Up via Shared Memory
* conv filter used here is identity
* ```stencil.cu```. Standard CUDA conv
* ```stencil_shared.cu```. CUDA conv with shared memory
	* Each thread reads the vector elements from global memory. Same elements are read multiple times by thread. Overlapping operations. Load into pre-shared memory
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
	

#### Result
* 20x faster on GPU vs CPU
* 60x faster on GPU vs CPU if using shared memory
* Further optimization possible
