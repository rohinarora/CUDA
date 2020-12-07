#include <stdio.h>
#include <stdlib.h>

#define RADIUS 8
#define BLOCK_SIZE 512
//conv filter used here is identity
__global__ void convBaseline(double *in, double *out, int vector_size) {
	// each thread computes result for 1 element of output conv
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; //global index
	if (gindex < vector_size){
		double result = 0.0;
		for (int offset = -RADIUS ; offset <= RADIUS ; ++offset)
			result += (gindex + offset >= 0 && gindex + offset < vector_size) ? in[gindex + offset] : 0.0;
		out[gindex] = result; 
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

	
	double *in_cpu         = new double [vector_size]; // CPU Struct
	double *out_cpu        = new double [vector_size];
	double *out_gpu_on_cpu = new double [vector_size]; // Copy GPU result to CPU

	printf("Initializing input arrays.\n");
	for (int i = 0; i < vector_size; i++) { // fill the arrays 'a' and 'b' on the CPU
		in_cpu[i] = (rand()%100)*cos(i);
		out_cpu[i] = 0.0;
		out_gpu_on_cpu[i] = 0.0;
	}
	
	double *in_gpu; // GPU pointers
	double *out_gpu;
	cudaMalloc (&in_gpu, vector_size*sizeof(double));
	cudaMalloc (&out_gpu, vector_size*sizeof(double));

	cudaMemcpy (in_gpu, in_cpu, vector_size*sizeof(double), cudaMemcpyHostToDevice);

	printf("Running sequential job.\n");
	cudaEventRecord(start,0);

	for (int i = 0; i < vector_size; ++i) { // Calculate C in the CPU
		for (int offset = -RADIUS ; offset <= RADIUS ; ++offset)
		   out_cpu[i] += (i + offset >= 0 && i + offset < vector_size) ? in_cpu[i + offset] : 0.0;
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\tSequential Job Time: %.2f ms\n", time);

	printf("Running parallel job.\n");

	cudaEventRecord(start,0);

	convBaseline<<<grid_size, BLOCK_SIZE>>>(in_gpu, out_gpu, vector_size);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job Time: %.2f ms\n", time);

	cudaMemcpy (out_gpu_on_cpu, out_gpu, vector_size*sizeof(double), cudaMemcpyDeviceToHost);
	
	int error = 0;
	for (int i = 0; i < vector_size; i++) {
		if (out_cpu[i] != out_gpu_on_cpu[i]){
			error = 1;
			printf( "Mistake at element %d\n", i);
			int start = (i-RADIUS<0)?0:i-RADIUS;
			int end = (i+RADIUS>vector_size)?vector_size:i+RADIUS;
			for (int offset = start ; offset <= end ; offset++)
                printf( "index = %d \tin = %.5lf \tout GPU = %.5lf \tCPU %.5lf\n", offset, 
															in_cpu[offset], 
															out_gpu_on_cpu[offset], 
															out_cpu[offset] );    
		}
		if (error) break; 
	}

	if (error == 0){
		printf ("Correct result. No errors were found.\n");
	}

	free (in_cpu); //free GPU
	free (out_cpu);
	free (out_gpu_on_cpu);
	cudaFree (in_gpu); //free GPU
	cudaFree (out_gpu);

	return 0;
}
