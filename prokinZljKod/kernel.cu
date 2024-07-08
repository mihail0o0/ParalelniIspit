#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 256
#define SHMEM 256

struct Point
{
    float x;
    float y;
};

struct Distance
{
    float dist;
    int index;
};

__global__ void min_reduction(Distance *g_distances, int n)
{
    __shared__ Distance *distances;
    distances = (Distance *)malloc(blockDim.x * 2);
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    distances[threadIdx.x] = g_distances[i];
    distances[threadIdx.x + blockDim.x] = g_distances[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            Distance l = distances[threadIdx.x];
            Distance r = distances[threadIdx.x + s];
            Distance min = r.dist < l.dist ? r : l;
            distances[threadIdx.x] = min;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        Distance min = distances[0];
        Distance tmp = g_distances[i];
        g_distances[i] = min;
        g_distances[min.index] = tmp;
    }
}

__global__ void compute_distances(Point *points, Distance *distances, int n, Point point)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        float dx = points[i].x - point.x;
        float dy = points[i].y - point.y;
        float dist = sqrt(pow(dx, 2) + pow(dy, 2));

        distances[i].dist = dist;
        distances[i].index = i;
    }
}

#define N (1 << 13)

int main(int argc, char **argv)
{
    Point point;
    point.x = 5;
    point.y = 10;

    Point *host_points;
    Point *dev_points;
    Distance *dev_distances;

    host_points = (Point *)malloc(N * sizeof(Point));

    for (int i = 0; i < N; i++)
    {
        host_points[i].x = i;
        host_points[i].y = 0;
    }

    cudaMalloc(&dev_points, N * sizeof(Point));
    cudaMalloc(&dev_distances, N * sizeof(Distance));

    cudaMemcpy(dev_points, host_points, N * sizeof(Point), cudaMemcpyHostToDevice);

    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_distances<<<GRID_SIZE, BLOCK_SIZE>>>(dev_points, dev_distances, N, point);

    Distance *host_distances = (Distance *)malloc(N * sizeof(Distance));
    cudaMemcpy(host_distances, dev_distances, N * sizeof(Distance), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << host_distances[i].dist << " ";
    std::cout << "\n";

    for (int i = 0; i < 10; i++)
    {
        min_reduction<<<GRID_SIZE, BLOCK_SIZE / 2>>>(dev_distances + i, N - i);
        min_reduction<<<1, BLOCK_SIZE / 2>>>(dev_distances + i, GRID_SIZE);

        Distance min;
        cudaMemcpy(&min, dev_distances + i, 1 * sizeof(Distance), cudaMemcpyDeviceToHost);

        std::cout << host_points[min.index].x << " - " << host_points[min.index].y << " : " << min.dist << "\n";
    }
}