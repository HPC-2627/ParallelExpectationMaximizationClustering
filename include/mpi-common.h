#ifndef MPI_COMMON_H
#define MPI_COMMON_H

/*
 * mpi-common.h
 *
 * Helpers shared by ALL the MPI + OpenMP versions (diagonal and full, baseline-MPI-OpenMP,
 * opt1 and opt2). Header-only (static inline) and NOT in src/common/, because src/common/ is
 * also compiled by baseline-sequential (diagonal and full), which is built without MPI. Include it only from
 * the MPI versions.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* ----------------------------------------------------------------------------------------
 * Abort the WHOLE job from one rank.
 *
 * If a rank simply returned after an error, the others would wait forever in the next
 * collective call. MPI_Abort terminates all the ranks.
 * Used like printf:  ABORT("cannot read %s", path);
 * ---------------------------------------------------------------------------------------- */
#define ABORT(...)                                  \
    do                                              \
    {                                               \
        fprintf(stderr, "error: " __VA_ARGS__);     \
        fprintf(stderr, "\n");                      \
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);    \
    } while (0)

/* ----------------------------------------------------------------------------------------
 * Block distribution of the rows: how many rows the rank owns and where its block starts.
 *
 * N_total rows over P ranks: every rank gets N_total / P rows and the first N_total % P ranks
 * get one more. The blocks are contiguous and cover all the rows with no overlap.
 * Computed locally by every rank (it only needs N_total, P and its own rank): no communication.
 * ---------------------------------------------------------------------------------------- */
static inline void block_of_rank(long N_total /* in */, int P /* in */, int rank /* in */,
                                 long *first_row /* out */, long *n_local /* out */)
{
    long q = N_total / P, rem = N_total % P;
    *n_local = q + (rank < rem ? 1 : 0);
    *first_row = rank * q + (rank < rem ? rank : rem);
}

/* Maximum over the ranks of a time: the parallel program is as slow as its slowest rank.
 * The result is valid only on rank 0. */
static inline double max_over_ranks(double local /* in */)
{
    double global = 0.0;
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global;
}

/* Gathers one double from every rank on rank 0: all[r] = value of rank r. 'all' must have room
 * for P values on rank 0 (it is not touched on the other ranks, it can be NULL there). Used to
 * log the compute time of every rank (load imbalance). */
static inline void gather_over_ranks(double local /* in */, double *all /* out, rank 0 */)
{
    MPI_Gather(&local, 1, MPI_DOUBLE, all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

#endif
