#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define SIZE 1024

using namespace std;

__global__ void multiplyMatrices(int *d_v1, int *d_v2, int *d_vsum, int n)
{
}

int main()
{
    size_t bytes = SIZE * SIZE * sizeof(int);

    int *a, *b, *rez;
    int *d_a, *d_b, *d_rez;

    a = (int *)malloc(bytes);
    b = (int *)malloc(bytes);
    rez = (int *)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_rez, bytes);

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            int index = i * SIZE + j;
            a[index] = index;
            b[index] = index;
        }
    }

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 16;
    int gridSize = (SIZE + blockSize - 1 / blockSize);

    dim3 grids(gridSize, gridSize);
    dim3 threads(blockSize, blockSize);

    multiplyMatrices<<<grids, threads>>>(d_a, d_b, d_rez, SIZE * SIZE);

    return 0;
}