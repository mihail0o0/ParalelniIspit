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
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    int currSum = 0;

    if (row < n && col < n)
    {
        for (int k = 0; k < n; k++)
        {
            currSum += d_a[row * n + k] * d_b[n * k + col];
        }

        d_rez[row * n + col] = currSum;
    }
}

__host__ void verify(int *a, int *b, int *rez, int n);
int main()
{
    int *a;
    int *b;
    int *c;

    int *d_a;
    int *d_b;
    int *d_c;

    size_t bytes = SIZE * SIZE * sizeof(int);

    a = (int *)malloc(bytes);
    b = (int *)malloc(bytes);
    c = (int *)malloc(bytes);

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            int index = i * SIZE + j + 1;
            a[index] = index;
            b[index] = index;
        }
    }

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    int threadNum = 32;
    int blockNum = (SIZE + threadNum - 1) / threadNum;
    dim3 threads(threadNum, threadNum);
    dim3 blocks(blockNum, threadNum);

    auto start = chrono::high_resolution_clock::now();

    transpose<<<threads, blocks>>>(d_a, d_b, d_c, SIZE);
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Elapsed: " << duration.count() << "ms." << endl;

    string doVerification;
    cout << "Calculation finished, verify result? Y(es) N(o): ";

    cin >> doVerification;

    if (doVerification == "Y" || doVerification == "y")
    {
        start = chrono::high_resolution_clock::now();
        verify(a, b, c, SIZE);
        end = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "CPU time elapsed: " << duration.count() << "ms." << endl;
    }
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