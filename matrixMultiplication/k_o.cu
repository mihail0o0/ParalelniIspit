#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#define SIZE (1 << 10)

using namespace std;

__global__ void transpose(int *d_a, int *d_b, int *d_rez, int n)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int currSum = 0;

    if (row < n && col < n)
    {
        for (int k = 0; k < n; k++)
        {
            currSum += d_a[row * n + k] * d_b[k * n + col];
        }
        d_rez[row * n + col] = currSum;
    }
}

__host__ void verify(int *a, int *b, int *rez, int n);
int main()
{
    cout << SIZE << " x " << SIZE << " elements matrix." << endl;
    size_t bytes = SIZE * SIZE * sizeof(int);

    int *a, *b, *rez;
    int *d_a, *d_b, *d_rez;

    a = (int *)malloc(bytes);
    b = (int *)malloc(bytes);
    rez = (int *)malloc(bytes);

    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_rez, bytes);

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

    int blockSize = 32;
    int gridSize = (SIZE + blockSize - 1) / blockSize;

    dim3 grids(gridSize, gridSize);
    dim3 threads(blockSize, blockSize);

    cout << "starting work..." << endl;
    auto start = chrono::high_resolution_clock::now();

    transpose<<<grids, threads>>>(d_a, d_b, d_rez, SIZE);
    cudaMemcpy(rez, d_rez, bytes, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < SIZE; i++)
    // {
    //     for (int j = 0; j < SIZE; j++)
    //     {
    //         cout << rez[i * SIZE + j] << " ";
    //     }
    //     cout << endl;
    // }

    auto end = chrono::high_resolution_clock::now();

    string doVerification;
    cout << "Calculation finished, verify result? Y(es) N(o): ";

    cin >> doVerification;

    if (doVerification == "Y" || doVerification == "y")
    {
    verify(a, b, rez, SIZE);
    }

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Elapsed time: " << duration.count() << "ms." << endl;

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

    cout << mat[10] << " " << rez[10] << endl;

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