# Build rules.
#
#   make              builds the sequential version  (bin/PEMC-baseline-sequential-diagonal)
#   make sequential   same as above
#   make clean        removes bin/
#
# The code shared by all the versions is in src/common/*.c (the headers are in include/).
# It does not use MPI, so the same rule compiles it both for the sequential version (gcc) and,
# later, for the MPI + OpenMP versions (mpicc -fopenmp): their targets must be added below.

CC       ?= gcc
CFLAGS   ?= -O3 -std=gnu11 -Wall -Wextra
INCLUDES  = -Iinclude
LIBS      = -lm

BIN_DIR    = bin
COMMON_SRC = $(wildcard src/common/*.c)
HEADERS    = $(wildcard include/*.h)

.PHONY: all sequential clean

all: sequential

sequential: $(BIN_DIR)/PEMC-baseline-sequential-diagonal $(BIN_DIR)/PEMC-baseline-sequential-full

# It is rebuilt if the main source, a shared file or a header changes
$(BIN_DIR)/PEMC-baseline-sequential-diagonal: src/PEMC-baseline-sequential-diagonal.c $(COMMON_SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) src/PEMC-baseline-sequential-diagonal.c $(COMMON_SRC) -o $@ $(LIBS)

$(BIN_DIR)/PEMC-baseline-sequential-full: src/PEMC-baseline-sequential-full.c $(COMMON_SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) src/PEMC-baseline-sequential-full.c $(COMMON_SRC) -o $@ $(LIBS)

# MPI + OpenMP versions: not part of "all", so a machine without MPI can still build the
# sequential ones. On the cluster: module load MPICH/3.4.2-GCC-10.3.0, then make mpi-omp.
# On macOS Apple clang has no OpenMP: use a GCC from Homebrew, e.g.
#   OMPI_CC=gcc-16 make mpi-omp        (Open MPI)   or   MPICH_CC=gcc-16 make mpi-omp   (MPICH)
MPICC    ?= mpicc
MPIFLAGS  = $(CFLAGS) -fopenmp

.PHONY: mpi-omp
mpi-omp: $(BIN_DIR)/PEMC-baseline-MPI-OpenMP-diagonal

$(BIN_DIR)/PEMC-baseline-MPI-OpenMP-diagonal: src/PEMC-baseline-MPI-OpenMP-diagonal.c $(COMMON_SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(MPIFLAGS) $(INCLUDES) src/PEMC-baseline-MPI-OpenMP-diagonal.c $(COMMON_SRC) -o $@ $(LIBS)

clean:
	rm -rf $(BIN_DIR)
