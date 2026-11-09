# Build rules.
#
#   make              builds the sequential versions  (bin/PEMC-baseline-sequential-{diagonal,full})
#   make sequential   same as above
#   make mpi-omp      builds the MPI + OpenMP versions (needs mpicc)
#   make clean        removes bin/
#
# Layout: src/diagonal/ and src/full/ hold one main file per version (diagonal or full covariance),
# src/common/ the code shared by all of them (no MPI), include/ the headers. Every executable goes
# flat in bin/ with the same name as its source file (PEMC-xxx.c -> bin/PEMC-xxx).
# To add a version, append its name to SEQ_BINS or MPI_BINS: the pattern rules find the source.

CC       ?= gcc
CFLAGS   ?= -O3 -std=gnu11 -Wall -Wextra
INCLUDES  = -Iinclude
LIBS      = -lm

BIN_DIR    = bin
COMMON_SRC = $(wildcard src/common/*.c)
HEADERS    = $(wildcard include/*.h)

# make looks for PEMC-xxx.c in these folders
vpath %.c src/diagonal src/full

SEQ_BINS = $(BIN_DIR)/PEMC-baseline-sequential-diagonal \
           $(BIN_DIR)/PEMC-baseline-sequential-full

# MPI + OpenMP versions. Not yet written (empty files): opt1/opt2-diagonal, opt1/opt2-full.
# Add them here when they are ready.
MPI_BINS = $(BIN_DIR)/PEMC-baseline-MPI-OpenMP-diagonal \
           $(BIN_DIR)/PEMC-baseline-MPI-OpenMP-full

.PHONY: all sequential mpi-omp clean

all: sequential

sequential: $(SEQ_BINS)

# On the cluster: module load MPICH/3.4.2-GCC-10.3.0, then make mpi-omp.
# On macOS Apple clang has no OpenMP: use a GCC from Homebrew, e.g.
#   OMPI_CC=gcc-16 make mpi-omp        (Open MPI)   or   MPICH_CC=gcc-16 make mpi-omp   (MPICH)
MPICC    ?= mpicc
MPIFLAGS  = $(CFLAGS) -fopenmp

mpi-omp: $(MPI_BINS)

# Rebuilt if the main source, a shared file or a header changes.
$(SEQ_BINS): $(BIN_DIR)/%: %.c $(COMMON_SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_SRC) -o $@ $(LIBS)

$(MPI_BINS): $(BIN_DIR)/%: %.c $(COMMON_SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(MPIFLAGS) $(INCLUDES) $< $(COMMON_SRC) -o $@ $(LIBS)

clean:
	rm -rf $(BIN_DIR)
