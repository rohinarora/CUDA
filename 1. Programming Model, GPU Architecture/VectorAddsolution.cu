#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c, int vector_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // This if statement is added in case we have more threads executing than \
  // number of elements in the vectors. 
  // All threads in wrap execute same code
  // We don't want threads with tid > vector_size to run this code 
  // tid > vector_size -> this condition will be true for certain threads in *last* wrap
  if (tid < vector_size) {
    c[tid] = a[tid] + b[tid];
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s vector_size block_size\n", argv[0]);
    return 1;
  }
  int vector_size = atoi(argv[1]);
  int block_size = atoi(argv[2]);
  int grid_size = ((vector_size - 1) / block_size) + 1;
  cudaSetDevice(0);
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int *a = new int[vector_size];
  int *b = new int[vector_size];
  int *c_cpu = new int[vector_size];
  int *c_gpu = new int[vector_size];
  // Pointers in GPU memory
  int *dev_a;
  int *dev_b;
  int *dev_c;
  printf("Initializing input arrays.\n");
  for (int i = 0; i < vector_size; i++) {
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }
  // CPU Calculation
  printf("Running sequential job.\n");
  cudaEventRecord(start, 0);
  for (int i = 0; i < vector_size; i++) {
    c_cpu[i] = a[i] + b[i];
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("\tSequential Job Time: %.2f ms\n", time);
  // allocate the memory on the GPU
  cudaMalloc(&dev_a, sizeof(int) * vector_size);
  cudaMalloc(&dev_b, sizeof(int) * vector_size);
  cudaMalloc(&dev_c, sizeof(int) * vector_size);
  cudaMemcpy(dev_a, a, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
  // GPU Calculation
  printf("Running parallel job.\n");
  cudaEventRecord(start, 0);
  add<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, vector_size); // call the kernel
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("\tParallel Job Time: %.2f ms\n", time);
  cudaMemcpy(c_gpu, dev_c, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);
  // compare the results
  int error = 0;
  for (int i = 0; i < vector_size; i++) {
    if (c_cpu[i] != c_gpu[i]) {
      error = 1;
      printf("Error starting element %d, %d != %d\n", i, c_gpu[i], c_cpu[i]);
    }
    if (error)
      break;
  }
  if (error == 0) {
    printf("Correct result. No errors were found.\n");
  }
  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  return 0;
}
