* conv filter used here is identity
* stencil.cu. Simple CUDA conv
* each thread reads the vector elements from global memory. preloading vector seen by a block into shared memory will save time. (prevents global memory fetch every time. many elements are read multiple times by thread. overlapping operations)
```
(base) [arora.roh@c2138 HOL4]$ ./stencil 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1923.53 ms
Running parallel job.
	Parallel Job Time: 50.76 ms
Correct result. No errors were found.
```
* stencil_shared.cu. CUDA conv with shared memory
```
(base) [arora.roh@c2138 HOL4]$ ./stencil_shared 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 1848.85 ms
Running parallel job.
	Parallel Job Time: 14.66 ms
Correct result. No errors were found.
```
	* further optimization possible. overlapping computations (if kernel was non identity, repeated multiplications)
* 20x faster on GPU vs CPU
* 60x faster on GPU vs CPU if using shared memory
* shared memory can be seen as a cache memory that you control what is being stored in it
