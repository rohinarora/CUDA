#### sess05.pdf
* Slide 6.
  * Random accesses are very hard to improve
  * Generally applications don’t have random accesses
  * Locality
* Slide 8
  * Memory model in GPU
* Slide 11. Memory Coalescing
  * Suppose 1 wrap -> 32 threads working on data very close spatially in memory. GPU will do a single read for them (exploiting spatial locality). Instead of 32 reads.
* Slide 12 good. Hence save in row major
* Slide 13 good.
* Slide 14-17. Registers.
* Slide 18. Local memory
* Slide 19-31 Shared memory
  * Shared memory is much much faster than local memory. Local memory shown here is physically outside SM
  * Shared memory and L1 cache are both physically local to SM. Both at same hierarchy. (slide 22)
  * Shared memory is local to a SM. Hence threads in a block can access the same shared memory space. Threads in a block will stay in a SM
  * Need to synchronize shared memory access, and must be taken care by programmer. This memory is being accessed by all the threads in a block
  * \__syncthreads() slows the execution a bit, as some wraps will be waiting for others to complete (to achieve sync)
* Slide 22. Shared memory vs L1 cache.
  * Shared memory is seen by programmer. L1 is not. L1 is only managed by the hardware
* Slide 23. Using shared memory in program.
* Slide 25-31. broadcast/collision etc
* Slide 32, 33. Cache
  * "Non-programmable memory", hardware concept
* Slide 34-37. Constant Memory
  * Host has to initialize it. Not kernel code
  * If threads do lot of reads to same address- use constant memory. This value can then be cached into L1 for life of program, and give good timing. Like value of Pi. This can also broadcast. (Slide 36)
  * Slide 35. cudaMemcpyToSymbol - special memcopy just for constant memory
* Slide 38-41. Texture Memory
  * Not sure how Nvidia implements
  * Slide 40. Z ordering helps in convolution.
  * Striding, Kernel filter size effect Z order
  * Slide 41. Using texture memory. 2D array normally stored as row major. Binding/unbinding might be creating Z order
* Slide 42-44 Good summary. Most important slides.
* Slide 45-49 Synchronization, atomic operations
  * Use in histogram
  * CUDA provides atomic functions on shared memory and global memory
  * atomic arithmetic/bitwise operations
* Slide 50-53
  * Granularities of synchronization and ways to deal with them
  
#### Questions
  * Slide 12. Math not clear
