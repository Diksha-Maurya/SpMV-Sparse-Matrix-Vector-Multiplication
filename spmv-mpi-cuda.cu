// spmv-mpi-cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
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

// Usage message.
void usage(int argc, char** argv) {
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be a real-valued sparse matrix in the MatrixMarket file format.\n");
}

__global__ void spmv_coo_kernel(const int local_nnz, 
                                const int *d_rows, 
                                const int *d_cols, 
                                const float *d_vals, 
                                const float *d_x, 
                                float *d_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_nnz) {
        atomicAdd(&d_y[d_rows[i]], d_vals[i] * d_x[d_cols[i]]);
    }
}

double benchmark_coo_spmv_cuda(coo_matrix *coo, float* x, float* y) {
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
    int local_nnz = lend - lstart;

    int *d_rows_local, *d_cols_local;
    float *d_vals_local;
    cudaMalloc((void**)&d_rows_local, local_nnz * sizeof(int));
    cudaMalloc((void**)&d_cols_local, local_nnz * sizeof(int));
    cudaMalloc((void**)&d_vals_local, local_nnz * sizeof(float));

    cudaMemcpy(d_rows_local, coo->rows + lstart, local_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols_local, coo->cols + lstart, local_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals_local, coo->vals + lstart, local_nnz * sizeof(float), cudaMemcpyHostToDevice);

    int num_cols = coo->num_cols;
    int num_rows = coo->num_rows;
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (local_nnz + blockSize - 1) / blockSize;

    timer time_one_iteration;
    timer_start(&time_one_iteration);
    spmv_coo_kernel<<<gridSize, blockSize>>>(local_nnz, d_rows_local, d_cols_local, d_vals_local, d_x, d_y);
    cudaDeviceSynchronize();
    double local_estimated_time = seconds_elapsed(&time_one_iteration);
    double estimated_time;
    MPI_Reduce(&local_estimated_time, &estimated_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int num_iterations;
    if (rank == 0) {
        if (estimated_time == 0)
            num_iterations = MAX_ITER;
        else
            num_iterations = min(MAX_ITER, max(MIN_ITER, (int)(TIME_LIMIT / estimated_time)));
        printf("\tPerforming %d iterations\n", num_iterations);
    }
    MPI_Bcast(&num_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    timer t;
    timer_start(&t);
    for (int j = 0; j < num_iterations; j++) {
        spmv_coo_kernel<<<gridSize, blockSize>>>(local_nnz, d_rows_local, d_cols_local, d_vals_local, d_x, d_y);
        cudaDeviceSynchronize();
    }
    double local_msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double msec_per_iteration;
    MPI_Reduce(&local_msec_per_iteration, &msec_per_iteration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double sec_per_iteration = msec_per_iteration / 1000.0;
        double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) num_nonzeros / sec_per_iteration) / 1e9;
        double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
        printf("\tbenchmarking COO-SpMV MPI+CUDA: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", 
               msec_per_iteration, GFLOPs, GBYTEs);
    }

    cudaMemcpy(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rows_local);
    cudaFree(d_cols_local);
    cudaFree(d_vals_local);
    cudaFree(d_x);
    cudaFree(d_y);

    return msec_per_iteration;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_id = rank % deviceCount;
    cudaSetDevice(device_id);

    if (get_arg(argc, argv, "help") != NULL) {
        if (rank == 0) usage(argc, argv);
        MPI_Finalize();
        return 0;
    }
    char *mm_filename = NULL;
    if (argc == 1) {
        if (rank == 0) printf("Give a MatrixMarket file.\n");
        MPI_Finalize();
        return -1;
    } else {
        mm_filename = argv[1];
    }

    coo_matrix coo;
    if (rank == 0) {
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

    if (rank == 0) {
        srand(13);
        for (int i = 0; i < coo.num_nonzeros; i++) {
            coo.vals[i] = 1.0f - 2.0f * (rand() / (RAND_MAX + 1.0f));
        }
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", 
               mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
        fflush(stdout);

        FILE *fp = fopen("test_COO", "w");
        fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
        fprintf(fp, "coo.rows:\n");
        for (int i = 0; i < coo.num_nonzeros; i++) {
            fprintf(fp, "%d  ", coo.rows[i]);
        }
        fprintf(fp, "\n\n");
        fprintf(fp, "coo.cols:\n");
        for (int i = 0; i < coo.num_nonzeros; i++) {
            fprintf(fp, "%d  ", coo.cols[i]);
        }
        fprintf(fp, "\n\n");
        fprintf(fp, "coo.vals:\n");
        for (int i = 0; i < coo.num_nonzeros; i++) {
            fprintf(fp, "%f  ", coo.vals[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }

    MPI_Bcast(coo.vals, coo.num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float *x = (float*)malloc(coo.num_cols * sizeof(float));
    float *y = (float*)malloc(coo.num_rows * sizeof(float));
    if (rank == 0) {
        for (int i = 0; i < coo.num_cols; i++) {
            x[i] = rand() / (RAND_MAX + 1.0f);
        }
    }
    MPI_Bcast(x, coo.num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < coo.num_rows; i++) {
        y[i] = 0.0f;
    }

    double msec_per_iteration = benchmark_coo_spmv_cuda(&coo, x, y);

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, y, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(y, NULL, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        FILE *fp = fopen("test_x", "w");
        for (int i = 0; i < coo.num_cols; i++) {
            fprintf(fp, "%f\n", x[i]);
        }
        fclose(fp);
        fp = fopen("test_y", "w");
        for (int i = 0; i < coo.num_rows; i++) {
            fprintf(fp, "%f\n", y[i]);
        }
        fclose(fp);
        delete_coo_matrix(&coo);
    } else {
        free(coo.rows);
        free(coo.cols);
        free(coo.vals);
    }
    free(x);
    free(y);
    MPI_Finalize();
    return 0;
}
