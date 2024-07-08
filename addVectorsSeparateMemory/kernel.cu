#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

#define SIZE (1 << 10)
#define threads 32

using namespace std;

__global__ void addTwoArrays(int *d_v1, int *d_v2, int *d_vsum, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    while (index < n)
    {
        d_vsum[index] = d_v1[index] + d_v2[index];
        index += blockDim.x * gridDim.x;
    }
}

int main()
{
    // int *v1;
    // int *v2;
    // int *vsum;

    // int *d_v1;
    // int *d_v2;
    // int *d_vsum;

    // size_t bytes = SIZE * sizeof(int);

    // v1 = (int *)malloc(bytes);
    // v2 = (int *)malloc(bytes);
    // vsum = (int *)malloc(bytes);

    // cudaMalloc(&d_v1, bytes);
    // cudaMalloc(&d_v2, bytes);
    // cudaMalloc(&d_vsum, bytes);

    // for (int i = 0; i < SIZE; i++)
    // {
    //     v1[i] = i;
    //     v2[i] = SIZE;
    // }

    // cudaMemcpy(d_v1, v1, bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_v2, v2, bytes, cudaMemcpyHostToDevice);
    // addTwoArrays<<<(SIZE + 127) / 128, 128>>>(d_v1, d_v2, d_vsum, SIZE);
    // cudaMemcpy(vsum, d_vsum, bytes, cudaMemcpyDeviceToHost);

    // cout << "rez: ";

    // for (int i = 0; i < SIZE; i++)
    // {
    //     cout << v1[i] << " " << v2[i] << " " << vsum[i] << endl;
    // }

    // cout << endl;

    // cudaFree(d_v1);
    // cudaFree(d_v2);
    // cudaFree(d_vsum);
    // free(v1);
    // free(v2);
    // free(vsum);

    // cudaDeviceReset();
    // return 0;

    int *v1;
    int *v2;
    int *rez;

    int *d_v1;
    int *d_v2;
    int *d_rez;

    size_t bytes = SIZE * sizeof(int);

    v1 = (int *)malloc(bytes);
    v2 = (int *)malloc(bytes);
    rez = (int *)malloc(bytes);

    for (int i = 0; i < SIZE; i++)
    {
        v1[i] = i;
        v2[i] = SIZE;
    }

    cudaMalloc(&d_v1, bytes);
    cudaMalloc(&d_v2, bytes);
    cudaMalloc(&d_rez, bytes);

    cudaMemcpy(d_v1, v1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, bytes, cudaMemcpyHostToDevice);

    int threadNum = 128;
    int blockNum = (SIZE + threadNum - 1) / threadNum;

    cout << "Starting..." << endl;
    auto start = std::chrono::high_resolution_clock::now();

    addTwoArrays<<<blockNum, threadNum>>>(d_v1, d_v2, d_rez, SIZE);

    cudaMemcpy(rez, d_rez, bytes, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Finished..." << endl;

    for (int i = 0; i < SIZE; i++)
    {
        cout << v1[i] << " " << v2[i] << ": " << rez[i] << endl;
    }

    cout << "------------------" << endl;
    cout << "Elapsed: " << duration.count() << "ms." << endl;

    free(v1);
    free(v2);
    free(rez);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_rez);

    cudaDeviceReset();

    return 0;
}