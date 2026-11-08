#!/bin/bash
# PBS launcher for bin/PEMC-baseline-MPI-OpenMP-diagonal (build first: make mpi-omp).
#
# Layout: NODES nodes x RANKS_PER_NODE MPI processes x THREADS OpenMP threads per process.
# ncpus must be RANKS_PER_NODE * THREADS (cores reserved on every node).
# Change the three numbers below and the three #PBS lines together. Submit from the repo root:
#   qsub src/script/PEMC-baseline-MPI-OpenMP-diagonal.sh
#PBS -l select=2:ncpus=8:mpiprocs=2:mem=4gb
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -q shortCPUQ
#PBS -N pemc-baseline

NODES=2
RANKS_PER_NODE=2
THREADS=4
DATASET=data/real/brainweb/brainweb.bin
K=12

cd "$PBS_O_WORKDIR" || exit 1
module load MPICH/3.4.2-GCC-10.3.0

export OMP_NUM_THREADS=$THREADS
# One thread per core, kept on the core of its rank (avoids migrations between runs)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Multi-node runs must go through the UCX wrapper (see CLAUDE.md / course slides)
mpiexec -n $((NODES * RANKS_PER_NODE)) ./mpi_wrapper.sh \
    ./bin/PEMC-baseline-MPI-OpenMP-diagonal -d "$DATASET" -k $K
