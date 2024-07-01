#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#define SIZE (1 << 13)

#define SHEMEMSIZE 16 * 16 * 4

struct Point
{
    float x;
    float y;
};

__device__ void findClosest(int *niz, int *sum, int n, int jump);

__device__ void calculateDistances(Point *niz, Point point, float *distances, int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int distX = point.x - niz[index].x;
    int distY = point.y - niz[index].y;
    distances[index] = sqrt((distX * distX) + (distY * distY));
}

__global__ void findClosestDots(Point *niz, Point point, float *distances, int n)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index == 0)
    {
        calculateDistances(niz, point, distances, n);

        for (int i = 0; i < 10; i++)
        {
            findClosest(niz + (sizeof(int) * i), distances, n - 1, blockDim.x);
            findClosest(niz, distances, n, 1);
        }
    }
}

__device__ void findClosest(Point *niz, float *distances, int n, int jump)
{
    for (int i = jump / 2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            int lIndex = threadIdx.x;
            int rIndex = threadIdx.x + i;

            if (distances[lIndex] > distances[rIndex])
            {
                Point pom = niz[lIndex];
                niz[lIndex] = niz[rIndex];
                niz[rIndex] = pom;
            }
        }
    }
}

__host__ void verify(int *a, int *b, int *rez, int n);
int main()
{
    Point *niz, *rez;
    Point *d_niz, *d_rez;
    Point point;
    float *distance;
    float *d_distance;

    size_t pointBytes = SIZE * sizeof(Point);
    size_t bytes = SIZE * sizeof(float);

    niz = (Point *)malloc(pointBytes);
    rez = (Point *)malloc(pointBytes);
    distance = (float *)malloc(bytes);

    cudaMalloc(&d_niz, pointBytes);
    cudaMalloc(&d_rez, pointBytes);
    cudaMalloc(&d_distance, bytes);

    int i;
    for (i = 0; i < SIZE; i++)
    {
        niz[i].x = i;
        niz[i].y = 0;
    }

    std::cout << "Unesi x i y: ";
    std::cin >> point.x >> point.y;

    cudaMemcpy(d_niz, niz, pointBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rez, rez, pointBytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (SIZE + blockSize - 1) / blockSize;

    std::cout << "starting work..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // calculateDistances<<<gridSize, blockSize>>>(d_niz, point, d_distance, SIZE);
    findClosestDots<<<gridSize, blockSize>>>(d_niz, point, d_distance, SIZE);
    cudaMemcpy(distance, d_distance, bytes, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Elapsed: " << duration.count() << "ms." << std::endl;

    free(niz);
    free(rez);
    cudaFree(d_niz);
    cudaFree(d_rez);

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