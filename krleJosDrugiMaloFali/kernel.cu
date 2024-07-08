#include <math.h>
#include <iostream>

#define N (1 << 4)

struct Dijagonala
{
    int value, i;
};

__global__ void elementi(int *A, Dijagonala *d)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * N)
    {
        d[i].value = A[i * N + i];
        d[i].i = i;
    }
}

int main()
{
    int *A = new int[N * N];
    Dijagonala *d = new Dijagonala[N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = rand() % 11 + 2;
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }

    int *dev_A;
    Dijagonala *dev_d;

    size_t bytesMatrix = N * N * sizeof(int);
    size_t bytesDiag = N * sizeof(int);

    cudaMalloc(&dev_A, bytesMatrix);
    cudaMalloc(&dev_d, bytesDiag);

    cudaMemcpy(dev_A, A, bytesMatrix, cudaMemcpyHostToDevice);

    int blockSize = 16;
    int gridSize = ((blockSize - 1 + (N * N)) / blockSize);
    elementi<<<gridSize, blockSize>>>(dev_A, dev_d);
    cudaMemcpy(d, dev_d, bytesDiag, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("%d, (%d)\n", d[i].value, d[i].i);
    }

    // printf("Min element: %d\n", el[0]);

    delete[] A;
    delete[] d;
    cudaFree(dev_A);
    cudaFree(dev_d);
    return 0;
}