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

sequential: $(BIN_DIR)/PEMC-baseline-sequential-diagonal

# It is rebuilt if the main source, a shared file or a header changes
$(BIN_DIR)/PEMC-baseline-sequential-diagonal: src/PEMC-baseline-sequential-diagonal.c $(COMMON_SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) src/PEMC-baseline-sequential-diagonal.c $(COMMON_SRC) -o $@ $(LIBS)

clean:
	rm -rf $(BIN_DIR)
