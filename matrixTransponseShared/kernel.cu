#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#define SIZE (1 << 10)
#define BLOCK_DIM 32

using namespace std;

__global__ void transpose(int *a, int *b, int n)
{
    __shared__ int sblock[BLOCK_DIM][BLOCK_DIM];

    int row = (blockDim.y * BLOCK_DIM) + threadIdx.y;
    int col = (blockDim.x * BLOCK_DIM) + threadIdx.x;

    int currSum = 0;

    if (row < n && col < n)
    {
        int index = row * n + col;
        sblock[threadIdx.x][threadIdx.y] = a[index];
    }

    __syncthreads();
}

__host__ void verify(int *a, int *b, int *rez, int n);
int main()
{
    int *a;
    int *b;

    int *d_a;
    int *d_b;

    size_t bytes = SIZE * SIZE * sizeof(int);

    a = (int *)malloc(bytes);
    b = (int *)malloc(bytes);

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            int index = i * SIZE + j + 1;
            a[index] = index;
        }
    }

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

    int threadNum = 32;
    int blockNum = (SIZE + threadNum - 1) / threadNum;
    dim3 threads(threadNum, threadNum);
    dim3 blocks(blockNum, threadNum);

    auto start = chrono::high_resolution_clock::now();

    transpose<<<threads, blocks>>>(d_a, d_b, SIZE);
    cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Elapsed: " << duration.count() << "ms." << endl;

    string doVerification;
    cout << "Calculation finished, verify result? Y(es) N(o): ";

    cin >> doVerification;

    if (doVerification == "Y" || doVerification == "y")
    {
        start = chrono::high_resolution_clock::now();
        verify(a, b, SIZE);
        end = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "CPU time elapsed: " << duration.count() << "ms." << endl;
    }
}

__host__ void verify(int *a, int *b, int n)
{
    int *mat;
    mat = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            assert(a[j * n + i] == b[i * n + j]);
        }
    }

    cout << "Sljakam ko djokovic po beton" << endl;
}