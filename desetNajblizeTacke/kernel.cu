#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#define SIZE (1 << 13)

#define SHEMEMSIZE 16 * 16 * 4

__device__ void findClosest(int *niz, int *sum, int n, int jump);
__global__ void findClosestDots(int *niz, int *sum, int n)
{
    findClosest(niz, sum, n, blockDim.x);
    __syncthreads();
    findClosest(sum, sum, (n + blockDim.x - 1 / blockDim.x), 1);
}

__device__ void findClosest(int *niz, int *sum, int n, int jump)
{
    int index = jump * blockIdx.x + threadIdx.x;

    __shared__ int partial[SHEMEMSIZE];

    partial[threadIdx.x] = niz[index];

    __syncthreads();

    for (int i = 1; i < jump + 1; i *= 2)
    {
        if (threadIdx.x % (i * 2) == 0)
        {
            partial[threadIdx.x] += partial[threadIdx.x + i];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        sum[blockIdx.x] = partial[0];
    }
}

__host__ void verify(int *a, int *b, int *rez, int n);
int main()
{
    int *niz, *sum;
    int *d_niz, *d_sum;

    size_t bytes = SIZE * sizeof(int);

    niz = (int *)malloc(bytes);
    sum = (int *)malloc(bytes);

    cudaMalloc(&d_niz, bytes);
    cudaMalloc(&d_sum, bytes);

    for (int i = 0; i < SIZE; i++)
    {
        niz[i] = 1;
    }

    cudaMemcpy(d_niz, niz, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (SIZE + blockSize - 1) / blockSize;

    std::cout << "starting work..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    findClosestDots<<<gridSize, blockSize>>>(d_niz, d_sum, SIZE);
    cudaMemcpy(sum, d_sum, bytes, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Suma je: " << sum[0] << std::endl;
    std::cout << "Elapsed: " << duration.count() << "ms." << std::endl;

    free(niz);
    free(sum);
    cudaFree(d_niz);
    cudaFree(d_sum);

    cudaDeviceReset();

    return 0;
}

// __host__ void verify(int *a, int *b, int *rez, int n)
// {
//     int *mat;
//     mat = (int *)malloc(n * n * sizeof(int));

//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             for (int k = 0; k < n; k++)
//             {
//                 mat[i * n + j] += a[i * n + k] * b[k * n + j];
//             }
//         }
//     }

//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             bool vazi = rez[i * n + j] == mat[i * n + j];

//             assert(vazi);
//         }
//     }

//     cout << "Sljakam ko djokovic po beton" << endl;
// }