* Slide 11. Memory Coalescing
  * Suppose 1 wrap -> 32 threads working on data very close spatially in memory. GPU will do a single read for them (exploiting spatial locality). Instead of 32 reads.
  * Slide 12 very good. Hence save in row major
  * Slide 13 good.
* Slide 16
* Slide 18
* Slide 19
  * Shared memory is much much faster than local memory. Local memory shown here is outside SM
* Slide 20, 21- redo
* Slide 22. Shared memory vs L1 cache.
  * Shared memory is seen by programmer. L1 is not
* Slide 23. Redo.
* Slide 25
* Slide 32
  * "Non-programmable memory", hardware concept
* If threads do lot of reads to same address- use constant memory. If nearby - use shared
* Z ordering helps in convolution.
  * striding, kernel filter size effect Z order
* Print slide 43, 44. Maybe redo.
* Slide 46
