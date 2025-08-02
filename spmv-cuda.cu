// spmv-cuda.cu
#include <stdio.h>
#include <stdlib.h>
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

void usage(int argc, char** argv) {
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be a real-valued sparse matrix in the MatrixMarket file format.\n");
}

__global__ void spmv_coo_kernel(const int num_nonzeros, 
                                const int *rows, 
                                const int *cols, 
                                const float *vals, 
                                const float *x, 
                                float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nonzeros) {
        atomicAdd(&y[rows[i]], vals[i] * x[cols[i]]);
    }
}

double benchmark_coo_spmv_cuda(coo_matrix *coo, float* x, float* y) {
    int num_nonzeros = coo->num_nonzeros;
    
    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    cudaMalloc((void**)&d_rows, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_cols, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_vals, num_nonzeros * sizeof(float));
    cudaMalloc((void**)&d_x, coo->num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, coo->num_rows * sizeof(float));

    cudaMemcpy(d_rows, coo->rows, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, coo->cols, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, coo->vals, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, coo->num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, coo->num_rows * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_nonzeros + blockSize - 1) / blockSize;

    timer time_one_iteration;
    timer_start(&time_one_iteration);
    
    spmv_coo_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
    cudaDeviceSynchronize();
    
    double estimated_time = seconds_elapsed(&time_one_iteration);
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
    }
    
    printf("\tPerforming %d iterations\n", num_iterations);

    timer t;
    timer_start(&t);
    for (int j = 0; j < num_iterations; j++) {
        spmv_coo_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
        cudaDeviceSynchronize();
    }
    double msec_total = milliseconds_elapsed(&t);
    double msec_per_iteration = msec_total / num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV CUDA: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", 
           msec_per_iteration, GFLOPs, GBYTEs);

    cudaMemcpy(y, d_y, coo->num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_y);

    return msec_per_iteration;
}

int main(int argc, char** argv) {
    if (get_arg(argc, argv, "help") != NULL) {
        usage(argc, argv);
        return 0;
    }
    if (argc < 2) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    }
    char *mm_filename = argv[1];

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    srand(13);
    for (int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0f - 2.0f * (rand() / (RAND_MAX + 1.0f));
    }
    
    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", 
           mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

    printf("Writing matrix in COO format to test_COO ...");
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
    printf("... done!\n");

    float *x = (float*)malloc(coo.num_cols * sizeof(float));
    float *y = (float*)malloc(coo.num_rows * sizeof(float));
    for (int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0f);
    }

    for (int i = 0; i < coo.num_rows; i++) {
        y[i] = 0.0f;
    }

    double msec_per_iteration = benchmark_coo_spmv_cuda(&coo, x, y);

    printf("Writing x and y vectors ...");
    FILE *fp_x = fopen("test_x", "w");
    for (int i = 0; i < coo.num_cols; i++) {
      fprintf(fp_x, "%f\n", x[i]);
    }
    fclose(fp_x);
    FILE *fp_y = fopen("test_y", "w");
    for (int i = 0; i < coo.num_rows; i++) {
      fprintf(fp_y, "%f\n", y[i]);
    }
    fclose(fp_y);
    printf("... done!\n");

    printf("Result vector y (first 10 entries):\n");
    for (int i = 0; i < (coo.num_rows < 10 ? coo.num_rows : 10); i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    free(x);
    free(y);
    delete_coo_matrix(&coo);

    return 0;
}
