#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{
	printf("Hello CUDA world \n");
}

// Begin Documentation
//  In this example, we are considering a grid with 64 threads as a 16 X 4 2D matrix. 
//  Inside this grid, we will have each thread block with 16 threads, arranged into a 2D matrix 
//  of 8 X 2. For the grid, we will have the thread blocks arranged in a 2D  2 X 2 grid.
//  When this program executes, it will execute 64 threads in the GPU, and will print
//  out "Hello World" 64 times.
// End Documentation

int main()
{
	int nx, ny;
	nx = 16;
	ny = 4;

	dim3 block(8, 2);
	dim3 grid(nx / block.x,ny / block.y);

	hello_cuda << < grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
