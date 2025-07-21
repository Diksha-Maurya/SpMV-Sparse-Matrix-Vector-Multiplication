CC=gcc
MPICC  = mpicc
NVCC = nvcc
#FLAG=-g -Wall
FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
LDFLAG=-O3
NVFLAGS  = -O3 -std=c++11 -I./include/ -diag-suppress 177,2464

OBJS=spmv.o mmio.o 
CUDA_OBJS = spmv-cuda.o mmio.o

OBJS       = spmv.o mmio.o
MPI_OBJS   = spmv-mpi.o mmio.o
OMP_OBJS   = spmv-omp.o mmio.o
HYBRID_OBJS    = spmv-hybrid.o mmio.o

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv: ${OBJS}
	${CC}  ${LDFLAG} -o $@ $^

spmv-mpi.o: spmv-mpi.c
	$(MPICC) $(FLAG) -c spmv-mpi.c -o spmv-mpi.o

spmv-omp.o: spmv-omp.c
	$(CC) $(FLAG) -fopenmp -c spmv-omp.c -o spmv-omp.o

spmv-hybrid.o: spmv-hybrid.c
	$(MPICC) $(FLAG) -fopenmp -c spmv-hybrid.c -o spmv-hybrid.o

spmv-mpi: $(MPI_OBJS)
	$(MPICC) $(LDFLAG) -o spmv-mpi $(MPI_OBJS)

spmv-omp: $(OMP_OBJS)
	$(CC) $(LDFLAG) -fopenmp -o spmv-omp $(OMP_OBJS)

spmv-hybrid: $(HYBRID_OBJS)
	$(MPICC) $(LDFLAG) -fopenmp -o spmv-hybrid $(HYBRID_OBJS)


spmv-cuda:spmv-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-cuda.cu -o $@

spmv-mpi-cuda: spmv-mpi-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-mpi-cuda.cu -ccbin mpicxx -Xcompiler "-DMPICH_SKIP_MPICXX" -o $@ -lmpi

%.o: %.c
	$(CC) -c $(FLAG) $< -o $@

%.o: %.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@

.PHONY:clean
clean: 
	find ./ -name "*.o" -delete
	rm spmv spmv-mpi spmv-omp spmv-hybrid spmv-cuda spmv-mpi-cuda