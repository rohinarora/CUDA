#include <stdio.h>
#include <stdlib.h>

__global__ void add( int *a, int *b, int *c, int vector_size ) { // must be pointers in GPU memory
    int tid = blockDim.x*blockIdx.x+threadIdx.x; //which thread is running
    if (tid < vector_size){   // All threads in wrap execute same code. We don't want threads with tid > vector_size to run this code. tid > vector_size -> this condition will be true for certain threads in *last* wrap
      c[tid] = a[tid]+b[tid];
    }
}

int main( int argc, char* argv[] ) { 
    if (argc != 3) {
        printf ("Usage: %s vector_size block_size\n", argv[0]);
        return 1;
    }
    int vector_size = atoi(argv[1]);
    int block_size  = atoi(argv[2]);
    int grid_size   = ((vector_size-1)/block_size) + 1;

    cudaSetDevice(0); // Set device that we will use for our cuda code
    cudaEvent_t start, stop; // Time Variables
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    int *a        = new int [vector_size];  // Input Arrays and variables
    int *b        = new int [vector_size]; 
    int *c_cpu    = new int [vector_size]; 
    int *c_gpu    = new int [vector_size];

    printf("Initializing input arrays.\n"); // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < vector_size; i++) {
        a[i] = rand()%10;
        b[i] = rand()%10;
    }

    printf("Running sequential job.\n"); // CPU Calculation
    cudaEventRecord(start,0);
    for (int i = 0; i < vector_size; i++) { // Calculate C in the CPU
            c_cpu[i] = a[i] + b[i];
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\tSequential Job Time: %.2f ms\n", time);

    int *dev_a; // Pointers in GPU memory
    int *dev_b;
    int *dev_c;

    cudaMalloc((void **)&dev_a, sizeof(int)*vector_size); // allocate the memory on the GPU
    cudaMalloc((void **)&dev_b, sizeof(int)*vector_size);
    cudaMalloc((void **)&dev_c, sizeof(int)*vector_size);

    cudaMemcpy(dev_a,a,sizeof(float)*vector_size,cudaMemcpyHostToDevice); // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_b,b,sizeof(float)*vector_size,cudaMemcpyHostToDevice);

    printf("Running parallel job.\n");
    cudaEventRecord(start,0);
    add<<<grid_size,block_size>>>(dev_a,dev_b,dev_c,vector_size); // GPU Calculation
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\tParallel Job Time: %.2f ms\n", time);

    cudaMemcpy(c_gpu,dev_c,sizeof(float)*vector_size,cudaMemcpyDeviceToHost); // copy the array 'c' back from the GPU to the CPU
    
    int error = 0; // compare the results
    for (int i = 0; i < vector_size; i++) {
        if (c_cpu[i] != c_gpu[i]){
            error = 1;
            printf( "Error starting element %d, %d != %d\n", i, c_gpu[i], c_cpu[i] );    
        }
        if (error) break; 
    }

    if (error == 0){
        printf ("Correct result. No errors were found.\n");
    }

    free (a); // free CPU data
    free (b);
    free (c_cpu);
    free (c_gpu);
    cudaFree(dev_a); // free the memory allocated on the GPU
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}