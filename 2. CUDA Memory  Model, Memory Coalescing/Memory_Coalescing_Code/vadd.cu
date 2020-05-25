/*
Baseline
*/
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE   512

typedef struct Data {
	double a;
	double b;
	double c;
	
} Data;

__global__ void add( Data *data, int vector_size ) {
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < vector_size){
		data[tid].c = data[tid].a + data[tid].b;
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

	Data *data_cpu        = new Data [vector_size];  //array of struct
	Data *data_gpu;  //gpu pointer
	Data *data_gpu_on_cpu = new Data [vector_size];  //copy back from GPU to CPU here

	printf("Initializing input arrays.\n");
	for (int i = 0; i < vector_size; i++) { // fill the arrays 'a' and 'b' on the CPU
		data_cpu[i].a = rand()*cos(i);
		data_cpu[i].b = rand()*sin(i);
		data_cpu[i].c = 0.0;
	}

	cudaMalloc (&data_gpu, vector_size*sizeof(Data));
	cudaMemcpy (data_gpu, data_cpu, vector_size*sizeof(Data), cudaMemcpyHostToDevice); // copy the input to the GPU

	printf("Running sequential job.\n");
	cudaEventRecord(start,0);

	for (int i = 0; i < vector_size; i++) {
			data_cpu[i].c = data_cpu[i].a + data_cpu[i].b; // Calculate C in the CPU
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

	cudaMemcpy (data_gpu_on_cpu, data_gpu, vector_size*sizeof(Data), cudaMemcpyDeviceToHost); // copy the GPU data to the CPU
	
	int error = 0;
	for (int i = 0; i < vector_size; i++) { // compare the results
		if (data_cpu[i].c != data_gpu_on_cpu[i].c){
			error = 1;
			printf( "Error starting element %d, %f != %f\n", i, data_gpu_on_cpu[i].c, data_cpu[i].c );    
		}
		if (error) break; 
	}

	if (error == 0){
		printf ("Correct result. No errors were found.\n");
	}

	free (data_cpu); // free CPU data
	free (data_gpu_on_cpu);
	cudaFree (data_gpu); // free the memory allocated on the GPU

	return 0;
}

