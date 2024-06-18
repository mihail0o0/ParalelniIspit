#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define SIZE 100
#define threads 32

using namespace std;

__global__ void addTwoArrays(int *d_v1, int *d_v2, int *d_vsum, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        d_vsum[index] = d_v1[index] + d_v2[index];
    }
}

int main()
{
    int id = cudaGetDevice(&id);

    size_t bytes = SIZE * sizeof(int);

    int *v1, *v2, *vsum;

    cudaMallocManaged(&v1, bytes);
    cudaMallocManaged(&v2, bytes);
    cudaMallocManaged(&vsum, bytes);

    for (int i = 0; i < SIZE; i++)
    {
        v1[i] = i;
        v2[i] = SIZE;
    }

    // mala optimizacija, podaci se prefetchuju u background 
    cudaMemPrefetchAsync(v1, bytes, id);
    cudaMemPrefetchAsync(v2, bytes, id);
    addTwoArrays<<<(SIZE + 127 / 128), 128>>>(v1, v2, vsum, SIZE);

    // cekamo da graficka zavrsi svoj poso
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(vsum, bytes, cudaCpuDeviceId);

    for (int i = 0; i < SIZE; i++)
    {
        cout << v1[i] << " " << v2[i] << " " << vsum[i] << endl;
    }

    return 0;
}
