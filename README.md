# SpMV-Sparse-Matrix-Vector-Multiplication

🧮 Sparse Matrix-Vector Multiplication (SpMV) - Parallel Implementations
Implemented and evaluated Sparse Matrix-Vector Multiplication (SpMV) using the Coordinate (COO) format, where the sparse matrix is represented via rows, cols, and vals arrays. The dense vectors x and y are stored as arrays.

This project focuses on improving performance of SpMV—a core operation in HPC and deep learning (especially post-pruning)—by addressing the challenges of irregular memory access patterns and sparse data representation.

🔧 Implementations
✅ Sequential Version (baseline)

⚙️ MPI Parallel Version – spmv-mpi.c

⚙️ OpenMP Multithreaded Version – spmv-omp.c

⚙️ Hybrid MPI + OpenMP Version – spmv-hybrid.c

📊 Dataset
Used real-world sparse matrices from the SuiteSparse Matrix Collection:

D6-6, dictionary28, Ga3As3H12, bfly, pkustk14, roadNet-CA
SuiteSparse: http://sparse.tamu.edu

✅ Correctness
Floating-point mismatches accepted as long as the integer part matches the sequential result.
