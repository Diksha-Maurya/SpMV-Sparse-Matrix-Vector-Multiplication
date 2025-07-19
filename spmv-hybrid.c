#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h> 
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y)
{
    int num_nonzeros = coo->num_nonzeros;
    int rank, size, lstart, lend;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int quotient_value = num_nonzeros / size;
    int remainder_value = num_nonzeros % size;

    if (rank < remainder_value) {
      lstart = rank * (quotient_value + 1);
      lend = lstart + (quotient_value + 1);
    } else {
      lstart = rank * quotient_value + remainder_value;
      lend = lstart + quotient_value;
    }

    timer time_one_iteration;
    timer_start(&time_one_iteration);
    #pragma omp parallel for schedule(static)
    for (int i = lstart; i < lend; i++){
        #pragma omp atomic
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }
    double local_estimated_time = seconds_elapsed(&time_one_iteration);
    double estimated_time;

    MPI_Reduce(&local_estimated_time, &estimated_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    int num_iterations;
    if (rank == 0){
        if (estimated_time == 0)
            num_iterations = MAX_ITER;
        else
            num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) );
        printf("\tPerforming %d iterations\n", num_iterations);
    }

    MPI_Bcast(&num_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    timer t;
    timer_start(&t);
    for(int j = 0; j < num_iterations; j++)
        #pragma omp parallel for schedule(static)
        for (int i = lstart; i < lend; i++){
            #pragma omp atomic
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    double local_msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double msec_per_iteration;
    MPI_Reduce(&local_msec_per_iteration, &msec_per_iteration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0){
        double sec_per_iteration = msec_per_iteration / 1000.0;
        double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
        double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
        printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs);
    }
    return msec_per_iteration;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (get_arg(argc, argv, "help") != NULL){
        if (rank == 0) usage(argc, argv);
        MPI_Finalize();
        return 0;
    }
    char * mm_filename = NULL;
    if (argc == 1) {
        if (rank == 0) printf("Give a MatrixMarket file.\n");
        MPI_Finalize();
        return -1;
    } else 
        mm_filename = argv[1];
    coo_matrix coo;
    if (rank == 0){
        read_coo_matrix(&coo, mm_filename);
    }
    MPI_Bcast(&coo.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_nonzeros, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        coo.rows = (int*)malloc(coo.num_nonzeros * sizeof(int));
        coo.cols = (int*)malloc(coo.num_nonzeros * sizeof(int));
        coo.vals = (float*)malloc(coo.num_nonzeros * sizeof(float));
    }
    MPI_Bcast(coo.rows, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.cols, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.vals, coo.num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank == 0){
        srand(13);
        for(int i = 0; i < coo.num_nonzeros; i++) {
            coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
        }
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
        fflush(stdout);
    
    printf("Writing matrix in COO format to test_COO ...");
    FILE *fp = fopen("test_COO", "w");
    fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fprintf(fp, "coo.rows:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.rows[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.cols:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.cols[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.vals:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%f  ", coo.vals[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("... done!\n");
  }
    MPI_Bcast(coo.vals, coo.num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
    float * y = (float*)malloc(coo.num_rows * sizeof(float));
    if (rank == 0){
        for(int i = 0; i < coo.num_cols; i++) {
            x[i] = rand() / (RAND_MAX + 1.0);
        }
    }
    MPI_Bcast(x, coo.num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for(int i = 0; i < coo.num_rows; i++)
        y[i] = 0;
    double coo_gflops;
    coo_gflops = benchmark_coo_spmv(&coo, x, y);

    if (rank == 0){
        MPI_Reduce(MPI_IN_PLACE, y, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        FILE *fp = fopen("test_x", "w");
        for (int i=0; i<coo.num_cols; i++){
          fprintf(fp, "%f\n", x[i]);
        }
        fclose(fp);
        fp = fopen("test_y", "w");
        for (int i=0; i<coo.num_rows; i++){
          fprintf(fp, "%f\n", y[i]);
        }
        fclose(fp);
    } else {
        MPI_Reduce(y, NULL, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0){
        delete_coo_matrix(&coo);
        free(x);
        free(y);
    } else {
        free(x);
        free(y);
        free(coo.rows);
        free(coo.cols);
        free(coo.vals);
    }
    MPI_Finalize();
    return 0;
}
