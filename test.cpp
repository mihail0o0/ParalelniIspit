#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sstream>
#include <assert.h>


using namespace std;

void okt2000()
{

    // #pragma omp parallel for firstprivate(x)
    //     for (int k = 0; k < 100; k++)
    //     {
    //         z[k] = k + x;
    //         x = k;
    //     }

    //     for (int i = 0; i < 100; i++)
    //     {
    //         printf("%d \n", z[i]);
    //     }

    int z[100];
    int x = 1;
    int nizX[100];

#pragma omp parallel for firstprivate(x)
    for (int k = 0; k < 100; k++)
    {
        nizX[k] = k;
    }

#pragma omp for
    for (int k = 0; k < 100; k++)
    {
        z[k] = k + nizX[k];
    }

    for (int i = 0; i < 100; i++)
    {
        printf("%d \n", z[i]);
    }
}

void okt2023Vljd()
{
    omp_set_num_threads(3);

#pragma omp parallel
    {
#pragma omp task
        printf("Task1\n");

#pragma omp task
        printf("Task2\n");
    }
}

void sezdesetsest()
{
    int i;
    double sum;
    omp_set_num_threads(4);

    sum = 0.0;
#pragma omp parallel for private(sum)
    for (i = 1; i <= 4; i++)
    {
        sum = sum + i;
        printf("Sum is: %lf\n", sum);
    }
}

//  Napisati program na OpenMP kojim se vrši paralelno izračunavanje proizvoda bez
// korišćenja odredbe reduction i sa korišćenjem ove odredbe. Predvideti štampanje
// rezultata.

#define gxSize 50
long long int globalX[gxSize];

void initializeX()
{
    for (int i = 0; i < gxSize; i++)
    {
        globalX[i] = 2;
    }
}

long long int sezdesetosamsareduction()
{
    long long mul = 1;

#pragma omp parallel for reduction(* : mul)
    for (int i = 0; i < gxSize; i++)
    {
        mul *= globalX[i];
    }

    printf("%lld", mul);

    return mul;
}

long long int sezdesetosambezreduction()
{
    long long mul = 1;
    long long localProd = 1;

#pragma omp parallel firstprivate(localProd) shared(mul)
    {
#pragma omp for
        for (int i = 0; i < gxSize; i++)
        {
            localProd *= globalX[i];
        }

#pragma omp atomic
        mul *= localProd;
    }

    printf("%lld", mul);

    return mul;
}

void checkSezdesetosam(long long int checkMul)
{
    long long int mul = 1;

    for (int i = 0; i < gxSize; i++)
    {
        mul *= globalX[i];
    }

    assert(mul == checkMul);
}

// x++;
// a = x + 2;
// b = a + 3;
// c++

void checkNajs(int rezX, int rezA, int rezB, int rezC);
void najs()
{
    int x = 0;
    int a = 0;
    int b = 0;
    int c = 0;
    int localX;

    omp_set_num_threads(4);
#pragma omp parallel shared(localX)
    {
#pragma omp single
        {
            x++;
        }

#pragma omp sections
        {
#pragma omp section
            a = x + 2;
#pragma omp section
            b = x + 5;
#pragma omp section
            c++;
        }
    }

    checkNajs(x, a, b, c);
    printf("%d, %d, %d, %d\n", x, a, b, c);
}

void checkNajs(int rezX, int rezA, int rezB, int rezC)
{
    int x = 0;
    int a = 0;
    int b = 0;
    int c = 0;

    x++;
    a = x + 2;
    b = a + 3;
    c++;

    printf("%d, %d, %d, %d\n", x, a, b, c);

    assert(x == rezX);
    assert(a == rezA);
    assert(b == rezB);
    assert(c == rezC);
}



int main()
{

    // okt2000();

    // okt2023Vljd();

    // sezdesetsest();

    // initializeX();
    // long long int rez = sezdesetosamsareduction();
    // long long int rez = sezdesetosambezreduction();
    // checkSezdesetosam(rez);

    // najs();
}