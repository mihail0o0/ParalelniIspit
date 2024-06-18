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
    int *v1;
    int *v2;
    int *vsum;

    int *d_v1;
    int *d_v2;
    int *d_vsum;

    size_t bytes = SIZE * sizeof(int);

    v1 = (int *)malloc(bytes);
    v2 = (int *)malloc(bytes);
    vsum = (int *)malloc(bytes);

    cudaMalloc(&d_v1, bytes);
    cudaMalloc(&d_v2, bytes);
    cudaMalloc(&d_vsum, bytes);

    for (int i = 0; i < SIZE; i++)
    {
        v1[i] = i;
        v2[i] = SIZE;
    }

    cudaMemcpy(d_v1, v1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, bytes, cudaMemcpyHostToDevice);
    addTwoArrays<<<(SIZE + 127) / 128, 128>>>(d_v1, d_v2, d_vsum, SIZE);
    cudaMemcpy(vsum, d_vsum, bytes, cudaMemcpyDeviceToHost);

    cout << "rez: ";

    for (int i = 0; i < SIZE; i++)
    {
        cout << v1[i] << " " << v2[i] << " " << vsum[i] << endl;
    }

    cout << endl;

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_vsum);
    free(v1);
    free(v2);
    free(vsum);

    cudaDeviceReset();
    return 0;
}