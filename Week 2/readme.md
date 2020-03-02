* **Hardware and software abstractions**
* slide 6
  * CUDA Driver API. Low level, more control
  * CUDA Runtime API. Higher level; implemented on top of the driver API. This repo uses CUDA runtime.
* slide 8- about cuda kernel
  * kernel is the operation that is actual code inside the nested loops
* Slide 10. Flow of a CUDA program
* slide 11- first use cudaMalloc. Allocate memory on GPU
  * cudaMemcpy has **const** void* src pointer. don't want to modify src data
* slide 13.
  * Grid has blocks. Blocks have threads. Threads run the kernel code
  * Blocks can be 1D, 2D or 3D.
  * Slide 13 shows 2D blocks
  * We see here 6 blocks each with 15 threads. Total 90 threads
* CUDA core is much simpler than a CPU core. Hence can have thousands of CUDA cores on a GPU chip
* Thread is a software abstraction for CUDA core. Threads "map" on to CUDA core. But can have more threads than CUDA cores for a SM, with extra threads in waiting state
* Slide 14. Important
  * A Kernel call creates a Grid.
  * All the blocks in the same grid contain the same number of threads.
  * The threads of a block can be indexed using 1 Dimension (x), 2 Dimensions (x,y) or 3 Dimensions indexes (x,y,z)
* ![](images/1.png)
* Slide 13. GridDim.X=3, GridDim.Y=2 blockDim.y=3. blockDim.x=5. If blockDim.z not defined, but its implicitly assigned as 1. 15 threads in the block. In general, number of threads in a block= (blockDim.x)\*(blockDim.y)\*(blockDim.z).
* Grid can be 3D. 3rd dimension has some small limit.
* Slide 15. In the end want to define block and grids so as to optimally utilize the GPU hardware at hand
  * dim3 is a struct in C that has x,y,z
* Slide 16
* Slide 20. Matrix multiplication in C.
* Slide 23. Can also compile C files with clang, cuda files with nvcc, and link everything later with clang
* Each SMx has 192 cuda cores. With 15 SMx we get access to 15*192 cuda cores
* SMx= Streaming Multiprocessors
* "Wrap" is a hardware concept. Not available to programmer. Managed by hardware.
  * Scheduler chunks the blocks into wraps- groups of 32 threads. These wraps are executed in the same SMx
* Block is a hardware and software concept.
  * Blocks are executed on same SMx (share L1 cache)
* Slide 28. Good
* Slide 29. Excellent
* Slide 30. 4 blocks. these 4 blocks can be scheduled on any SMx.
  * 256 threads within the block must run on same SMx. A SMx has 192 cuda cores. Hence some threads will be waiting while first 192 complete (scheduled in chunks of 32 threads, aka wraps)
* Minimum blocks i must have to have all the SMx running?
  * Total number of SMx
* Slide 28/29. Wrap on same SMx. Block on same SMx. Wrap is just a way to schedule block on to a SMx.
* Slide 30
  * Will occupy just 4/15 SMx at best at one time
* Slide 33. Wraps are 1D. Hardware concept
* Slide 34
  * If block size=33, 2 wraps. Both wraps on same SMx. 2nd wrap underutilized (1/32 cores being used in 2nd wrap)
* Slide 36. But SM has 192 cores. So 6 wraps at a time. The rest 58 wrap are in pending. 64 wraps are max in a SM.  This is running + pending + idle
* Slide 42. Good
* cudaMalloc has pointer to pointer as input argument. Not sure why
* compile VectorAddsolution.cu
  * run ./add 10000 2048
    * runs but doesn't give error !. only gives testing error.  _add some explanation here_ use wrapper_gpu_error. **todo**
  * profiling. **todo**
  * see lab once again incase you missed anything. **todo**
  * save exec.bash present in cluster **todo**
  * see Programming_Model_Execution_Model2 again. **todo**
  * take other source files Nico has **todo**
* Questions
* Slide 9. What do you mean by async call?
* Slide 11. Why void** in cudaMalloc
* Slide 13. What ifs. Answer to 4 questions
* Slide 19. What do you mean by "It has asynchronous behavior"?
