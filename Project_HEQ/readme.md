ssh rohin@didakt.me -p 6060
rohin@heq363
* create a branch from master. helper scripts. on gitlab
* In March 12, during the lab time, each group should present 2 slides about what were the optimizations attempted.
* make some graphs or come up with a way to show our results better.
* histogram update has to be by atomic operation
* https://www.youtube.com/watch?v=KOVUTeUNsh8
* https://www.youtube.com/watch?v=GWCB3pKi2ko
* https://www.youtube.com/watch?v=WuVyG4pg9xQ
* Host to Device Memory Transfers can be improved by using Pinned Memory and Unified Memory
* Use ideas from Part 3 (The Four Algorithms). Prefix sum for CDF
* Only modify the CUDA file
• Use shared memory
• Optimize the histogram using local histograms and then updating globally
• Process multiple pixels per thread
• Use texture memory
• Incorporate pinned memory (covered next week)
• Use streams for more parallelization (covered next week)
• For more information on histogram equalization:
∙ http://www.tutorialspoint.com/dip/Histogram_Equalization.htm
∙ https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
• Animation explaining Histogram Equalization by Telkon University
∙ https://www.youtube.com/watch?v=PD5d7EKYLcA
