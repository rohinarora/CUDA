#include <stdio.h>
#include <stdlib.h>

__global__ void MatMulGPU( float *a, float *b, float *c, int row,int col, int k ) { // must be pointers in GPU memory
    int tidx = blockDim.x*blockIdx.x+threadIdx.x; 
    int tidy = blockDim.y*blockIdx.y+threadIdx.y; 
    // A Kernel maps to each element of final matrix C
    if ((tidx < col) && (tidy < row)){   // tid > vector_size -> this condition will be true for certain threads in *last* wrap
        float sum=0;
        for (int z=0;z<k;++z){
            sum+=a[tidy*k+z]*b[z*k+tidx];
        }
        c[tidy*col+tidx] = sum;
    }
}

int main( int argc, char* argv[] ) { 
    if (argc != 3) {
        printf ("Usage: %s vector_size block_size\n", argv[0]);
        return 1;
    }

    cudaSetDevice(0); // Set device that we will use for our cuda code
    cudaEvent_t start, stop; // Time Variables
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    int row=atoi(argv[1]);
    int col=atoi(argv[2]);
    int k=col;
    float *a,*b,*c,*test;
    a=(float*) malloc(row*k*sizeof(float)); //a is row*k matrix
    b=(float*) malloc(k*col*sizeof(float)); //b is k*col matrix
    c=(float*) malloc(row*col*sizeof(float)); //c is matmul(a,b) on GPU
    test=(float*) malloc(row*col*sizeof(float)); //test is matmul(a,b) on CPU

    printf("Initializing input arrays.\n"); // fill the arrays 'a' and 'b' on the CPU
    for (int i=0;i<row;++i){ //init a
        for (int j=0;j<k;++j){
            a[i*k+j]=rand()%10;
        }
    }
    for (int i=0;i<k;++i){ //init b
        for (int j=0;j<col;++j){
            b[i*col+j]=rand()%10;
        }
    }

    printf("Running sequential job.\n"); // CPU matmul
    cudaEventRecord(start,0);
    for (int i=0;i<row;++i){ // Calculate matmul on the CPU
        for (int j=0;j<col;++j){
            int sum=0;
            for (int w=0;w<k;++w){
                sum+=a[i*k+w]*b[w*col+j];
            }
            test[i*col+j]=sum;
        }
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\tSequential Job Time: %.2f ms\n", time);

    float *dev_a; // Pointers in GPU memory
    float *dev_b;
    float *dev_c;

    cudaMalloc((void **)&dev_a, row*k*sizeof(float)); // allocate the memory on the GPU
    cudaMalloc((void **)&dev_b, k*col*sizeof(float));
    cudaMalloc((void **)&dev_c, row*col*sizeof(float));

    cudaMemcpy(dev_a,a,row*k*sizeof(float),cudaMemcpyHostToDevice); // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_b,b,k*col*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block_size(16,16,1);
    dim3 grid_size((((col-1)/block_size.x) + 1),(((row-1)/block_size.y) + 1),1);

    printf("Running parallel job.\n");
    cudaEventRecord(start,0);
    
    MatMulGPU<<<grid_size,block_size>>>(dev_a,dev_b,dev_c,row,col,k); // GPU Calculation
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\tParallel Job Time: %.2f ms\n", time);

    cudaMemcpy(c,dev_c,row*col*sizeof(float),cudaMemcpyDeviceToHost); // copy the array 'c' back from the GPU to the CPU

    int error = 0; 
    for (int i=0;i<row;++i){ // compare the results
        for (int j=0;j<col;++j){
            if (test[i*col+j] != c[i*col+j]){
                error = 1;
                printf( "Error starting element %d, %d != %d\n", i, test[i*col+j], c[i*col+j] );    
            }
            if (error) break; 
        }
    }
    if (error == 0){
        printf ("Correct result. No errors were found.\n");
    }
    /*
    free (a); // free CPU data
    free (b);
    free (c_cpu);
    free (c_gpu);
    cudaFree(dev_a); // free the memory allocated on the GPU
    cudaFree(dev_b);
    cudaFree(dev_c);
*/
    return 0;
}