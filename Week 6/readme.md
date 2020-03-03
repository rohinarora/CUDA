* Make sure you have correct grid and block size. What to fully use the GPU hardware
* If you have less blocks and more SM's, hardware won't be fully utilized (1 block will run on 1 SM completely. It won't split across SMs). Want blocks to be >= SMs (generally)
* Global memory coalescing. Exploit spatial locality.
* Constant memory. All threads reading same memory
* Texture memory. Z shape/transform. Exploit spatial locality. Not sure how Nvidia does it
* Shared memory banking. 32 banks each of 4 bytes.
  * data shared among different threads
  * different threads accessing different data across banks. one go. same cycle.
  * many threads accessing same data-> broadcast
  * many threads accessing different data in same bank. serial.
* Want to run kernel as fast as possible.
* Moving data has a cost !
* DDR 4kb pages. Swap memory.
* slide 9. You should not over-allocate pinned memory. Eats main OS resources
* Unified memory
  * Normally memory has 2 copies. Host and Device. Both have to be managed. Slide 10. explicit copy
* Slide 16. Eg multiplication and addition together
* Slide 22. Maybe even break original problem into parts so that problem can be parallelized better
  * Use this idea when kernel time is comparable to data transfer time. Profile your code !
  * Done in Slide 23

* Slide 40. If no stream given, NULL stream is used by default
