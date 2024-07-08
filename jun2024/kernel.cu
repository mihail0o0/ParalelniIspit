#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#define SIZE (1 << 10)
#define PI 3.14

using namespace std;

struct Circle
{
    float x;
    float y;
    float r;
};

struct Point
{
    float area;
    int index;
};

#define SHMEMSIZE (128 * sizeof(Point))

__global__ void calculateAreas(Circle *circles, Point *areas, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        areas[tid].area = circles[tid].r * circles[tid].r * PI;
        areas[tid].index = tid;
    }
}

__global__ void findMin(Point *areas, Point *resultAreas)
{
    __shared__ Point mem[SHMEMSIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    mem[threadIdx.x] = areas[tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            int l = threadIdx.x;
            int r = threadIdx.x + s;

            if (mem[l].area < mem[r].area)
            {
                Point pom = mem[l];
                mem[l] = mem[r];
                mem[r] = pom;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        resultAreas[blockIdx.x] = mem[0];
    }
}

int main()
{
    Circle *circles;
    Point *areas;
    Point *resultAreas;

    Circle *d_circles;
    Point *d_areas;
    Point *d_resultAreas;

    size_t circleBytes = SIZE * sizeof(Circle);
    size_t areaBytes = SIZE * sizeof(Point);

    circles = (Circle *)malloc(circleBytes);
    areas = (Point *)malloc(areaBytes);
    resultAreas = (Point *)malloc(areaBytes);

    cudaMalloc(&d_circles, circleBytes);
    cudaMalloc(&d_areas, areaBytes);
    cudaMalloc(&d_resultAreas, areaBytes);

    for (int i = 0; i < SIZE; i++)
    {
        circles[i].x = i;
        circles[i].y = i;
        circles[i].r = (i * 0.1) + 10;
    }

    circles[20].r = 6;

    cudaMemcpy(d_circles, circles, circleBytes, cudaMemcpyHostToDevice);

    int threadNum = 128;
    int blockNum = (SIZE + threadNum - 1) / threadNum;
    calculateAreas<<<blockNum, threadNum>>>(d_circles, d_areas, SIZE);
    cudaMemcpy(areas, d_areas, areaBytes, cudaMemcpyDeviceToHost);

    int resultIdx = 0;
    for (int i = 0; i < 10; i++)
    {
        cudaMemcpy(d_areas, areas, areaBytes, cudaMemcpyHostToDevice);
        findMin<<<blockNum, threadNum>>>(d_areas, d_resultAreas);
        findMin<<<1, threadNum>>>(d_resultAreas, d_resultAreas);
        cudaMemcpy(resultAreas, d_resultAreas, areaBytes, cudaMemcpyDeviceToHost);

        resultIdx = resultAreas[0].index;
        cout << i << "[" << resultIdx << "] = x: " << circles[resultIdx].x << ", y: " << circles[resultIdx].y << ", radius: " << circles[resultIdx].r << endl;

        areas[resultIdx].area = 0;
    }

    free(circles);
    free(areas);
    free(resultAreas);
    cudaFree(d_circles);
    cudaFree(d_areas);
    cudaFree(d_resultAreas);

    cudaDeviceReset();

    return 0;
}