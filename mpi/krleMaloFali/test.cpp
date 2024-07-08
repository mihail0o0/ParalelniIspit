#include <string>
#include "iostream"
#include "mpi.h"
#include "math.h"

#define k 7
#define n 2

int main(int argc, char **argv)
{
	int size, rank, A[k][n], b[n], d[k];
	MPI_Datatype vrsta;
	MPI_Status status;
	int locA[k][n]; //??????

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Type_vector(n, 1, 1, MPI_INT, &vrsta);
	MPI_Type_commit(&vrsta);

	if (rank == 0)
	{
		printf("A: \n");
		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < n; j++)
			{
				A[i][j] = 2 * i + j;
				printf("%d ", A[i][j]);
			}
			printf("\n");
		}
		printf("\n");

		printf("b: \n");
		for (int i = 0; i < n; i++)
		{
			b[i] = 3 * i;
			printf("%d ", b[i]);
		}
		printf("\n");
		printf("\n");

		// MPI_Send(&A[0][0], 1, vrsta, 0, 0, MPI_COMM_WORLD);
		// MPI_Recv(&locA, 2 * pow(2, rank), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		for (int i = 0; i < n; i++)
		{
			locA[0][i] = A[0][i];
		}

		int a = 1;
		for (int i = 1; i < size; i++)
		{
			MPI_Send(&A[a][0], pow(2, i), vrsta, i, 0, MPI_COMM_WORLD);
			a += pow(2, i);
		}
	}

	if (rank != 0)
	{
		MPI_Recv(&locA, 2 * pow(2, rank), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	}

	MPI_Bcast(&b[0], n, MPI_INT, 0, MPI_COMM_WORLD);

	int loc_d[n];
	for (int i = 0; i < n; i++)
	{
		loc_d[i] = 0;
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < pow(2, rank); j++)
		{
			loc_d[i] += locA[j][i] * b[i];
		}
	}

	printf("loc_d %d: ", rank);
	for (int i = 0; i < k; i++)
	{
		printf("%d ", loc_d[i]);
		printf("\n");
	}
	printf("\n");

	printf("locA %d: \n", rank);
	for (int i = 0; i < pow(2, rank); i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%d ", locA[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	printf("loc_b %d: \n", rank);
	for (int i = 0; i < n; i++)
	{
		printf("%d ", b[i]);
		printf("\n");
	}
	printf("\n");
	printf("\n");

	MPI_Reduce(&loc_d, &d, n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("d: ");
		for (int i = 0; i < k; i++)
		{
			printf("%d ", d[i]);
			printf("\n");
		}
		printf("\n");
	}

	MPI_Finalize();
	return 0;
}