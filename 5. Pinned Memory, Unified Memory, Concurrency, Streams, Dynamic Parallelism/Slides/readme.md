* Make sure you have correct grid and block size. What to fully use the GPU hardware
* If you have less blocks and more SM's, hardware won't be fully utilized (1 block will run on 1 SM completely. It won't split across SMs). Want blocks to be >= SMs (generally)
* Global memory coalescing. Exploit spatial locality.
* Constant memory. All threads reading same memory
* Texture memory. Z shape/transform. Exploit spatial locality. Not sure how Nvidia does it
* Shared memory banking. Array of 128 bytes, discrete 32 memory elements each of 4 bytes in a row.
  * Shared among threads in a block. 1 for each SM
  * If all threads access same address, allows broadcast (1 cycle)
  * If different threads accessing different addresses in a row (in a bank), 1 cycle
  * Many threads accessing different data in same column but different rows (in a bank). serial
* Want to run kernel as fast as possible.
* Moving data has a cost !
* Host to Device Memory Transfers can be improved by using Pinned Memory and Unified Memory
* Slide 5. Ways to minimize the amount of time it takes to transfer data between host and device.
* Slide 7. To access something from host disk, first it gets loaded into DDR memory in form of 4kB pages. When DDR gets full, OS swaps pages and puts them back in disk in "swap". Don't want the data being transferred from host to device to be swapped as there are more applications running on the host. To prevent this Cuda driver :
  * Allocate a temporary page-locked block, or “pinned”
  * Copy host data to the pinned block
  * Transfer from pinned to device
  * Delete pinned block
  * **It ensures the data will always be in RAM while it is being transferred**
  * **All this has a cost.**
* Slide 8. Code for Pinned Memory
* Run CudaCheck to make sure memory was pinned
* slide 9. You should not over-allocate pinned memory. Eats main OS resources
* Pin the memory that is accessed frequently
* Slide 10, 11. Unified memory. *This is a convenience*
  * Normally memory has 2 copies. Host and Device. Both have to be managed.
* Slide 11. Driver figures out whether the pointer is on device or host and automatically does the transfer if needed
* Slide 13-15. Benefit of Unified memory
* [Are cuda kernel calls synchronous or asynchronous](https://stackoverflow.com/questions/8473617/are-cuda-kernel-calls-synchronous-or-asynchronous)
* Cuda memcopy synchronous by default for the host.
* Cuda calls synchronous for GPU by default. Everything is submitted to the GPU via "NULL" stream by default
* Alternate: Use CudaMemCopyAsync. Needs pinned memory (slide 17)
* Slide 16. CUDA streams
  * Overlap two independent async CUDA calls
  * Eg multiplication and addition together
* Slide 18. Ways to synchronize
* Slide 21,22.
  * Break down the original problem into smaller subproblems
  * Use this idea when kernel time is comparable to data transfer time. Profile your code !
  * Done in Slide 23
* Slide 24.
  * 2 ways to do streaming
  * Comparison of the 2 methods in different cases
* Slide 40. If no stream given, NULL stream is used by default
* For "default-stream per-thread", refer-
https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
* Threads in default stream have implicit dependency
* Slide 44
  * Dynamic Parallelism
  * Each thread can spawn more threads/kernels; spawn another block
* Slide 51: Synchronization in dynamic parallelism
* Slide 63. GPU fibonacci with dynamic parallelism
