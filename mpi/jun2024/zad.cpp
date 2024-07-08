#include <string>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <math.h>

struct MaxRez
{
    double value;
    int index;
};

using namespace std;

int main(int argc, char **argv)
{

    int worldRank;
    int worldSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    ifstream inputFile;
    int n;
    int m;
    size_t arrSize;
    size_t recvSize;
    int *inputArr;
    int *recvBuff;

    int c;
    int localSum = 0;
    double localMid = 0;
    double mid = 0;

    double M = 0;
    double localM = 0;

    double localMax;
    MaxRez writeInIndex;

    // citanje i scatter niza
    if (worldRank == 0)
    {
        inputFile.open("./niz.dat");

        if (!inputFile.good())
        {
            cerr << "file not good";
            return -1;
        }

        inputFile >> n;

        c = 10;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c, 1, MPI_INT, 0, MPI_COMM_WORLD);

    m = n / worldSize;
    arrSize = n * sizeof(int);
    recvSize = m * sizeof(int);
    inputArr = (int *)malloc(arrSize);
    recvBuff = (int *)malloc(recvSize);

    if (worldRank == 0)
    {
        int i = 0;
        while (inputFile.good() && i < n)
        {
            inputFile >> inputArr[i];
            i++;
        }

        inputFile.close();
    }

    MPI_Scatter(inputArr, m, MPI_INT, recvBuff, m, MPI_INT, 0, MPI_COMM_WORLD);

    // racunanje mida
    for (int i = 0; i < m; i++)
    {
        localMid += recvBuff[i];
    }

    MPI_Reduce(&localMid, &mid, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (worldRank == 0)
    {
        mid /= n;
    }

    MPI_Bcast(&mid, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (worldRank == 1)
    {
        // cout << mid;
    }

    // glavna racunica

    for (int i = 0; i < m; i++)
    {
        localM += (recvBuff[i] + mid) / c;
    }

    double maxValue;

    // upis rezultata
    // trazim index u koj zelim upisati rez
    MaxRez localMaxRez;
    localMaxRez.index = worldRank;
    localMaxRez.value = (double)recvBuff[0];

    for (int i = 1; i < m; i++)
    {
        if ((double)recvBuff[i] > localMaxRez.value)
        {
            localMaxRez.value = (double)recvBuff[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&localMaxRez, &writeInIndex, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Bcast(&writeInIndex, 1, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);

    // finalna redukcija
    MPI_Reduce(&localM, &M, 1, MPI_DOUBLE, MPI_SUM, writeInIndex.index, MPI_COMM_WORLD);

    if(worldRank == writeInIndex.index){
        cout << "p: " << writeInIndex.index << ", test: " << writeInIndex.value << ", rez: " << M << endl;
    }

    MPI_Finalize();
    return 0;
}