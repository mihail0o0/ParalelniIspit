#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#define SIZE 1024

using namespace std;

__global__ void multiplyMatrices(int *d_a, int *d_b, int *d_rez, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    if ((row < n) && (col < n))
    {
        for (int k = 0; k < n; k++)
        {
            temp_sum += d_a[row * n + k] * d_b[k * n + col];
        }

        d_rez[row * n + col] = temp_sum;
    }
}

__host__ void verify(int *a, int *b, int *rez, int n);
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
            a[index] = index + 1;
            b[index] = index + 1;
        }
    }

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 16;
    int gridSize = (SIZE + blockSize - 1 / blockSize);

    dim3 grids(gridSize, gridSize);
    dim3 threads(blockSize, blockSize);

    multiplyMatrices<<<grids, threads>>>(d_a, d_b, d_rez, SIZE);
    cudaMemcpy(rez, d_rez, bytes, cudaMemcpyDeviceToHost);

    verify(a, b, rez, SIZE);

    free(a);
    free(b);
    free(rez);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_rez);

    cudaDeviceReset();

    return 0;
}

__host__ void verify(int *a, int *b, int *rez, int n)
{
    int *mat;
    mat = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                mat[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    cout << mat[5] << " " << rez[5] << " " << a[5] << b[5] << endl;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            bool vazi = rez[i * n + j] == mat[i * n + j];

            assert(vazi);
            
        }
    }

    cout << "Sljakam ko djokovic po beton" << endl;
}