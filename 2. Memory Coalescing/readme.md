* convert array of struct to struct of arrays to aid memory coalescing
* memory coalescing decreases runtime to 50%
* vdd.cu ->  baseline
* vadd_coalesced.cu -> optimized
```
//baseline
(base) [arora.roh@c2138 HOL3]$ ./vadd 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 385.82 ms
Running parallel job.
	Parallel Job Time: 32.80 ms
Correct result. No errors were found.
```
```
//optimized
(base) [arora.roh@c2138 HOL3]$ ./vadd_coalesced 100000000
Initializing input arrays.
Running sequential job.
	Sequential Job Time: 393.90 ms
Running parallel job.
	Parallel Job Time: 16.19 ms
Correct result. No errors were found.
```
