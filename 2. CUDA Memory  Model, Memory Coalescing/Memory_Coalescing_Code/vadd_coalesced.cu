#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE   512

typedef struct Data {
	double *a;
	double *b;
	double *c;
	
} Data;

__global__ void add( Data data, int vector_size ) {
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < vector_size){
		data.c[tid] = data.a[tid] + data.b[tid];
	}
}

int main( int argc, char* argv[] ) { 
	if (argc != 2) {
		printf ("Usage: %s vector_size\n", argv[0]);
		return 1;
	}
	int vector_size = atoi(argv[1]);
	int grid_size   = ((vector_size-1)/BLOCK_SIZE) + 1;

	cudaSetDevice(0);
        
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	Data data_cpu; // CPU Struct
	data_cpu.a = new double [vector_size]; 
	data_cpu.b = new double [vector_size]; 
	data_cpu.c = new double [vector_size]; 
	Data data_gpu; // GPU pointer
	Data data_gpu_on_cpu;  //copy back from GPU to CPU here
	data_gpu_on_cpu.c = new double [vector_size]; 

	printf("Initializing input arrays.\n");
	for (int i = 0; i < vector_size; i++) { // fill the arrays 'a' and 'b' on the CPU
		data_cpu.a[i] = rand()*cos(i);
		data_cpu.b[i] = rand()*sin(i);
		data_cpu.c[i] = 0.0;
	}
	
	cudaMalloc (&data_gpu.a, vector_size*sizeof(double));
	cudaMalloc (&data_gpu.b, vector_size*sizeof(double));
	cudaMalloc (&data_gpu.c, vector_size*sizeof(double));

	cudaMemcpy (data_gpu.a, data_cpu.a, vector_size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (data_gpu.b, data_cpu.b, vector_size*sizeof(double), cudaMemcpyHostToDevice);

	printf("Running sequential job.\n");
	cudaEventRecord(start,0);

	for (int i = 0; i < vector_size; i++) {
			data_cpu.c[i] = data_cpu.a[i] + data_cpu.b[i]; // Calculate C in the CPU
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\tSequential Job Time: %.2f ms\n", time);

	printf("Running parallel job.\n");

	cudaEventRecord(start,0);
	add<<<grid_size, BLOCK_SIZE>>>(data_gpu, vector_size);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job Time: %.2f ms\n", time);

	cudaMemcpy (data_gpu_on_cpu.c, data_gpu.c, vector_size*sizeof(double), cudaMemcpyDeviceToHost);
	
	int error = 0;
	for (int i = 0; i < vector_size; i++) {
		if (data_cpu.c[i] != data_gpu_on_cpu.c[i]){
			error = 1;
			printf( "Error starting element %d, %f != %f\n", i, data_gpu_on_cpu.c[i], data_cpu.c[i] );    
		}
		if (error) break; 
	}

	if (error == 0){
		printf ("Correct result. No errors were found.\n");
	}

	
	free (data_cpu.a); // free CPU data
	free (data_cpu.b);
	free (data_cpu.c);
	free (data_gpu_on_cpu.c);
	cudaFree (data_gpu.a);  // free GPU data
	cudaFree (data_gpu.b);
	cudaFree (data_gpu.c);

	return 0;
}

