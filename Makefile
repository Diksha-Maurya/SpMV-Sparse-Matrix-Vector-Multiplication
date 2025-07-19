CC     = gcc
MPICC  = mpicc

FLAG   = -O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
LDFLAG = -O3

OBJS       = spmv.o mmio.o
MPI_OBJS   = spmv-mpi.o mmio.o
OMP_OBJS   = spmv-omp.o mmio.o
HYBRID_OBJS    = spmv-hybrid.o mmio.o


%.o: %.c
	$(CC) $(FLAG) -c $< -o $@

spmv-mpi.o: spmv-mpi.c
	$(MPICC) $(FLAG) -c spmv-mpi.c -o spmv-mpi.o

spmv-omp.o: spmv-omp.c
	$(CC) $(FLAG) -fopenmp -c spmv-omp.c -o spmv-omp.o

spmv-hybrid.o: spmv-hybrid.c
	$(MPICC) $(FLAG) -fopenmp -c spmv-hybrid.c -o spmv-hybrid.o

spmv: $(OBJS)
	$(CC) $(LDFLAG) -o spmv $(OBJS)

spmv-mpi: $(MPI_OBJS)
	$(MPICC) $(LDFLAG) -o spmv-mpi $(MPI_OBJS)


spmv-omp: $(OMP_OBJS)
	$(CC) $(LDFLAG) -fopenmp -o spmv-omp $(OMP_OBJS)

spmv-hybrid: $(HYBRID_OBJS)
	$(MPICC) $(LDFLAG) -fopenmp -o spmv-hybrid $(HYBRID_OBJS)

.PHONY: clean
clean:
	find ./ -name "*.o" -delete
	rm -f spmv spmv-mpi spmv-omp spmv-hybrid
