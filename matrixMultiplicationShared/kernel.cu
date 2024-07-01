#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#define SIZE (1 << 15)

#define SHMEMSIZE (16 * 16 * 4)

using namespace std;

__global__ void sharedMultiplyMatrices(int *d_a, int *d_b, int *d_rez, int n, int tileSize)
{
    __shared__ int A[SHMEMSIZE];
    __shared__ int B[SHMEMSIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * tileSize + ty;
    int col = bx * tileSize + tx;

    for (int i = 0; i < n / tileSize; i++)
    {
        A[(ty * tileSize) + tx] = d_a[row * n + (i * tileSize + tx)];
        B[(ty * tileSize) + tx] = d_b[((i * tileSize + ty) * n) + col];
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

    int blockSize = 256;
    int gridSize = (SIZE + blockSize - 1 / blockSize);

    dim3 grids(gridSize, gridSize);
    dim3 threads(blockSize, blockSize);

    cout << "starting work..." << endl;
    auto start = chrono::high_resolution_clock::now();

    sharedMultiplyMatrices<<<grids, threads>>>(d_a, d_b, d_rez, SIZE);
    cudaMemcpy(rez, d_rez, bytes, cudaMemcpyDeviceToHost);

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