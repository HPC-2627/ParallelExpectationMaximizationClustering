/*
 * PEMC-opt1-MPI-OpenMP-full.c
 *
 * OPTIMIZATION 1 of the hybrid MPI + OpenMP EM, FULL covariance: ALL THE OPTIMIZATIONS OF THE
 * COMPUTATION. Same model, same formulas and same results as PEMC-baseline-MPI-OpenMP-full.c (read
 * it first); what changes is HOW the work inside a rank is organized: algorithm, loop order and
 * memory access. The communication is the one of the baseline on purpose (two MPI_Allreduce per
 * iteration): optimizing it is the subject of Optimization 2 (PEMC-opt2-MPI-OpenMP-full.c).
 * Points (1)-(3) are the same idea as PEMC-opt1-MPI-OpenMP-diagonal.c; point (4), the blocking,
 * exists only in the full model, whose cost per row (O(K*D^2)) makes it compute-bound.
 *
 * ========================================================================================
 * WHAT CHANGES WITH RESPECT TO THE BASELINE
 * ========================================================================================
 * (1) FUSION OF E-STEP AND M-STEP
 *     baseline: E-step loop  : reads X, writes the responsibilities r (n_local*K doubles)
 *               M-step loop  : reads X and r again, accumulates Nk, sum r*x, sum r*x*x^T
 *     opt1    : ONE pass over the rows. The responsibilities of a row (of a block of rows, see (4))
 *               are used immediately to update Nk, sx, sxx while the rows are still in the cache.
 *               The matrix r does not exist any more and X is read ONCE per iteration instead of
 *               twice (pem_bytes(fused = 1)). The labels (argmax of the responsibilities) are stored
 *               during the same pass.
 *
 * (2) ONE PARALLEL REGION PER ITERATION
 *     baseline: two "parallel for" per iteration (E-step, M-step), and the K Cholesky
 *               factorizations (K*D^3/6 multiply-adds) done by the master thread alone while the
 *               other threads wait (a serial part, Amdahl).
 *     opt1    : a single "#pragma omp parallel" with the K factorizations (split among the threads,
 *               one cluster each: they are independent), the loop over the rows (E+M fused) and the
 *               reductions of ll and of the sums. One fork/join per iteration.
 *
 * (3) NO FALSE SHARING ON THE PER-THREAD BUFFERS
 *     baseline: the per-thread buffers are slices of one malloc'ed array, aligned only to 16 bytes:
 *               the slice of a thread can share a 64-byte cache line with the slice of its neighbour,
 *               and every write invalidates the line in the cache of the other.
 *     opt1    : the per-thread buffers are allocated ONCE, aligned to 64 bytes, and every slice is a
 *               multiple of 64 bytes: no cache line is shared by two threads.
 *
 * (4) BLOCKS OF B ROWS, INNER LOOP OVER THE ROWS
 *     With (1)-(3) only, the fused pass reaches ~1.2 FLOP per cycle per core (synth_rare_p16,
 *     D = 50, ~28 GFLOP/s on 8 cores of the Xeon 6252N, arithmetic intensity ~78 FLOP/byte): it is
 *     compute-bound and the core waits instead of computing, because, processing ONE row at a time,
 *       (a) the forward substitution  t -= L[a][b] * z[b]  is a chain of DEPENDENT subtractions
 *           (each needs the previous t): the compiler cannot vectorize it (reordering floating-point
 *           additions changes the result), so it runs at ~one subtraction every 4 cycles;
 *       (b) the K Cholesky factors L_k are re-read for EVERY row and each value is used for ONE
 *           multiply (matrix-vector operation, no reuse in the registers);
 *       (c) the sums sxx (K*D*D doubles, 120 KB for K = 6, D = 50: larger than the 32 KB L1 cache)
 *           are read and written for EVERY row, while their value is needed only at the end.
 *     The independent work exists (the other ROWS), but in the outer loop, too far for the
 *     out-of-order window of the CPU. So every thread copies a block of B rows TRANSPOSED into its
 *     own buffer Xt (D x B: component a of the B rows contiguous) and runs, on the whole block:
 *       Mahalanobis_block      forward substitution of TILE = 8 rows at a time: the subtraction
 *                              chain of every row is unchanged, but the 8 chains are independent
 *                              and advance together in SIMD registers (with -O3 and no -march:
 *                              SSE2, 2 doubles per register, so 4 registers; one AVX-512
 *                              register with -march=cascadelake); each L[a][b] is loaded
 *                              once and used for 8 rows                                -> (a), (b)
 *       Responsibilities_block log-sum-exp, exp, label and log-likelihood of every row
 *       Accumulate_block       sxx[a][b] += sum_i r_ik x_ia x_ib accumulated in a register over the
 *                              B rows and written ONCE per block instead of once per row; the inner
 *                              loop over the rows is a contiguous SIMD reduction         -> (c)
 *     It is the step from matrix-vector operations (BLAS level 2: TRSV, SYR) to matrix-matrix ones
 *     (BLAS level 3: TRSM, SYRK), inspired by the matrix formulation with cache blocking of
 *     W. Kwedlo, FedCSIS 2023 (doc/papers_and_references.md #4.2), written here by hand (no BLAS
 *     library). Unlike Kwedlo (Cholesky of the inverse, TRMM, blocks sized for the L3 cache) we
 *     factorize Sigma itself, solve by forward substitution and size the blocks for L1/L2.
 *     The OpenMP loop runs over the blocks (schedule static).
 *
 *     Effect of (4) alone, measured on the cluster in a previous campaign (fused pass (1)-(3)
 *     without blocks vs with blocks, B = 64, 1 exclusive node, 8 cores; that version is in the git
 *     history, commit c9fadde): ~1.7x on the large datasets, 2.1x for D = 10, 1.7x for D = 50,
 *     1.05x on brainweb (D = 1: nothing to block, the cost is exp); 28 -> 49 GFLOP/s for D = 50.
 *
 *     Block size B (option -b, default 64): the per-thread buffers are ~(2*D + 2*K) * B doubles
 *     (~57 KB for D = 50, K = 6, B = 64): they must stay close to the core (L1/L2 cache, 1 MB of L2
 *     per core on the 6252N). Small B = little reuse of the sums; large B = buffers out of the
 *     cache. Local sweep (Apple M5, 4 threads, synth_rare_p01 / synth_clean_p01, time per iteration):
 *         B =  8: 166 / 9.4 ms   32: 149 / 8.5   64: 148 / 8.0   128: 175 / 8.4   512: 246 / 9.9
 *         (fused pass without blocks: 309 / 14.4 ms)
 *     B = 8 already gives most of the gain (the independent chains of (a)); beyond 128 the buffers
 *     leave the cache. To be repeated on the cluster (BLOCK=... src/script/pemc-submit.sh full opt1,
 *     logs in results/logs/block/, field "block" of the log).
 *
 * ========================================================================================
 * WHY THE RESULT DOES NOT CHANGE
 * ========================================================================================
 *   - Mahalanobis distance: for every row the operations are the SAME as in the baseline and in the
 *     same order (x - mu, the subtractions over b in increasing order, the DIVISION by L[a][a], not
 *     a multiplication by its inverse): the value is bitwise identical;
 *   - responsibilities and labels: same code as the baseline, row by row;
 *   - sums Nk, sx, sxx: the same products (r_ik * x_ia) * x_ib, added in a different order (within
 *     the block, then into the private copy of the thread): differences of rounding only (relative
 *     ~1e-12), as between two layouts of ranks x threads. The number of iterations and the final
 *     log-likelihood are the same as the baseline (|delta ll| / |ll| < 1e-8).
 *
 * CONVERGENCE AND THE SUMS OF THE LAST ITERATION
 *   ll must be known (summed over the ranks) BEFORE deciding whether to do the M-step. With the
 *   fusion the local sums are already computed by then. The order is therefore:
 *       fused pass -> Allreduce(ll) -> converged?  yes: stop (sums discarded, no second Allreduce)
 *                                                  no : Allreduce(sums) -> update parameters
 *   The only extra work with respect to the baseline is the accumulation of the sums in the last
 *   pass (thrown away): one pass out of all those performed.
 *
 * The compiler flags are those of all the other versions (-O3, no -march): SSE2 vectors of 2
 * doubles, no FMA. The gain therefore comes from the loop order, not from the flags. With
 * "make ARCH=-march=cascadelake" (all the versions) the vectors become wider: separate experiment.
 *
 * TIMES: the fused pass cannot be split in E and M, so it is reported in t_estep (Cholesky
 * included, as in the baseline); t_mstep is only the update of the parameters from the global sums
 * (K*D^2 operations); t_comm the MPI_Allreduce calls.
 *
 * Usage: see args_usage() (or run with -h).  Example:
 *   OMP_NUM_THREADS=4 mpiexec -n 2 bin/PEMC-opt1-MPI-OpenMP-full -d FILE.bin -k 4 -b 64
 */

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "args.h"
#include "common.h"
#include "dataset.h"
#include "eval.h"
#include "gmm.h"
#include "logger.h"
#include "mpi-common.h"
#include "perfmodel.h"

/* Name of the version: it ends up in the log and in the name of the default log file */
#define VERSION_NAME "opt1-MPI-OpenMP-full"

/* Rows per block when -b is not given (see the header for the choice) */
#define BLOCK_DEFAULT 64
/* Rows of the block processed together in the forward substitution: 8 doubles = one AVX-512
 * register, or 4 SSE2 registers of 2 doubles (the compiler splits it) */
#define TILE 8

/* Number of doubles rounded up to a multiple of 8 (= 64 bytes, one cache line) */
static size_t round_up8(size_t n /* in */)
{
    return (n + 7) & ~(size_t)7;
}

/* ----------------------------------------------------------------------------------------
 * Working memory of EM (of ONE rank). Allocated ONCE, reused at every iteration.
 * There is NO matrix of responsibilities (see (1)): only the labels of the last pass, and the
 * per-thread block buffers (see (3), (4)).
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    int *labels;  /* n_local: argmax_k r_ik of the LAST fused pass */
    double *cov;  /* K*D*D: covariance matrices Sigma_k (full, symmetric, row-major) */
    double *L;    /* K*D*D: Cholesky factors of Sigma_k (lower triangle used), rebuilt every pass */
    double *c;    /* K: part of the log-density that depends only on the cluster */
    double *acc;  /* K + K*D + K*D*D: the M-step sums, contiguous (one MPI_Allreduce), same layout
                     as the baseline:
                       acc[0 .. K-1]              Nk  = sum_i r_ik
                       acc[K .. K+K*D-1]          sx  = sum_i r_ik x_id        (index k*D + d)
                       acc[K+K*D .. end]          sxx = sum_i r_ik x_ia x_ib   (index k*D*D + a*D + b,
                                                  lower triangle b <= a only) */
    int B;        /* rows per block */
    int n_threads; /* threads for which tbuf has a slice */
    /* Per-thread block buffers: n_threads slices of 'slice' doubles, 64-byte aligned. Inside a
     * slice (each part rounded to a multiple of 8 doubles, so it starts on a cache line):
     *   Xt  D*B   the block, TRANSPOSED: Xt[a*B + i] = x_ia (component a of the B rows contiguous)
     *   Zt  D*B   solution of the forward substitution of the block; reused by Accumulate_block
     *             as RX[a*B + i] = r_ik x_ia
     *   LP  B*K   log(pi_k N(x_i|k)) of every row, then r_ik (row-major: the K values of a row are
     *             contiguous, as log_sum_exp wants)
     *   R   K*B   r_ik transposed: R[k*B + i] (the B rows of cluster k contiguous) */
    size_t off_Z, off_LP, off_R, slice;
    double *tbuf;
} Workspace;

static void workspace_alloc(Workspace *w /* out */, int n_local /* in */, int K /* in */,
                            int D /* in */, int B /* in */, int n_threads /* in */)
{
    w->labels = (int *)malloc((size_t)n_local * sizeof(int));
    w->cov = (double *)malloc((size_t)K * D * D * sizeof(double));
    w->L = (double *)malloc((size_t)K * D * D * sizeof(double));
    w->c = (double *)malloc((size_t)K * sizeof(double));
    w->acc = (double *)malloc((size_t)(K + K * D + (size_t)K * D * D) * sizeof(double));
    CHECK_ALLOC(w->labels);
    CHECK_ALLOC(w->cov);
    CHECK_ALLOC(w->L);
    CHECK_ALLOC(w->c);
    CHECK_ALLOC(w->acc);

    /* Layout of one slice (offsets in doubles, all multiples of 8) */
    w->B = B;
    w->n_threads = n_threads;
    const size_t db = round_up8((size_t)D * B), kb = round_up8((size_t)K * B);
    w->off_Z = db;
    w->off_LP = 2 * db;
    w->off_R = 2 * db + kb;
    w->slice = 2 * db + 2 * kb;
    void *mem = NULL;
    if (posix_memalign(&mem, 64, (size_t)n_threads * w->slice * sizeof(double)) != 0)
        mem = NULL;
    w->tbuf = (double *)mem;
    CHECK_ALLOC(w->tbuf);
}

static void workspace_free(Workspace *w /* in/out */)
{
    free(w->labels);
    free(w->cov);
    free(w->L);
    free(w->c);
    free(w->acc);
    free(w->tbuf);
}

/* ----------------------------------------------------------------------------------------
 * Block kernels. They work on ONE block of nb <= B rows, in the buffers of the calling thread,
 * and are called inside the parallel region (no OpenMP or MPI inside them).
 * ---------------------------------------------------------------------------------------- */

/* Copies the nb rows starting at X (row-major, D columns) into Xt, transposed: Xt[a*B + i] = x_ia */
static void Load_block_transposed(const double *restrict X /* in  */, int nb /* in */,
                                  int D /* in */, int B /* in */, double *restrict Xt /* out */)
{
    for (int i = 0; i < nb; i++)
        for (int a = 0; a < D; a++)
            Xt[(size_t)a * B + i] = X[(size_t)i * D + a];
}

/* Forward substitution L z = x - mu of vl <= TILE rows together (rows i0 .. i0+vl-1 of the
 * block), and squared Mahalanobis distance s[v] = |z|^2 of each. For every row the operations
 * are those of the baseline, in the same order; the loops over v are the vl INDEPENDENT rows, which go
 * in SIMD registers (4 SSE2 registers with the default flags, one with AVX-512). "static inline" with vl = TILE constant at the call: the compiler
 * specializes the full tiles (fixed length 8) and keeps a generic copy for the last one. */
static inline void Solve_tile(const double *restrict Xt /* in  */, const double *restrict mu /* in */,
                              const double *restrict L /* in  */, int i0 /* in */, int vl /* in */,
                              int D /* in */, int B /* in */, double *restrict Zt /* out */,
                              double *restrict s /* out: vl values */)
{
    for (int v = 0; v < vl; v++)
        s[v] = 0.0;
    for (int a = 0; a < D; a++)
    {
        const double *restrict La = L + (size_t)a * D;     /* row a of L_k */
        const double *restrict xa = Xt + (size_t)a * B + i0; /* x_ia of the vl rows */
        double t[TILE];
        for (int v = 0; v < vl; v++)
            t[v] = xa[v] - mu[a];
        for (int b = 0; b < a; b++)
        {
            const double l = La[b];                          /* ONE load, used for vl rows */
            const double *restrict zb = Zt + (size_t)b * B + i0;
#pragma omp simd
            for (int v = 0; v < vl; v++)
                t[v] -= l * zb[v];                           /* vl independent chains */
        }
        double *restrict za = Zt + (size_t)a * B + i0;
        const double d = La[a];
        for (int v = 0; v < vl; v++)
        {
            t[v] /= d; /* division as in the baseline (not * 1/d): same rounding */
            za[v] = t[v];
            s[v] += t[v] * t[v];
        }
    }
}

/* Log-density log(pi_k N(x_i | mu_k, Sigma_k)) = c_k - 0.5 |z|^2 of the nb rows of the block for
 * ONE cluster k, written in LP[i*K + k]. Zt is scratch (the z of the rows). */
static void Mahalanobis_block(const double *restrict Xt /* in */, const double *restrict mu /* in */,
                              const double *restrict L /* in */, double ck /* in */, int k /* in */,
                              int K /* in */, int nb /* in */, int D /* in */, int B /* in */,
                              double *restrict Zt /* scratch */, double *restrict LP /* out */)
{
    double s[TILE];
    for (int i0 = 0; i0 < nb; i0 += TILE)
    {
        const int vl = (nb - i0 < TILE) ? nb - i0 : TILE;
        if (vl == TILE)
            Solve_tile(Xt, mu, L, i0, TILE, D, B, Zt, s); /* full tile: length known */
        else
            Solve_tile(Xt, mu, L, i0, vl, D, B, Zt, s);   /* last, partial tile */
        for (int v = 0; v < vl; v++)
            LP[(size_t)(i0 + v) * K + k] = ck - 0.5 * s[v];
    }
}

/* Normalization of every row of the block (same code as the baseline, row by row): log-sum-exp,
 * responsibilities r_ik (overwriting LP), label = argmax_k, and the log-likelihood added to *ll.
 * The responsibilities are also written transposed in R[k*B + i] for Accumulate_block. */
static void Responsibilities_block(double *restrict LP /* in/out */, int nb /* in */, int K /* in */,
                                   int B /* in */, double *restrict R /* out */,
                                   int *restrict labels /* out: nb values */, double *ll /* in/out */)
{
    for (int i = 0; i < nb; i++)
    {
        double *restrict lp = LP + (size_t)i * K;
        const double lse = log_sum_exp(lp, K);
        *ll += lse;
        int best = 0;
        for (int k = 0; k < K; k++)
        {
            lp[k] = exp(lp[k] - lse);
            if (lp[k] > lp[best])
                best = k;
        }
        labels[i] = best;
        for (int k = 0; k < K; k++)
            R[(size_t)k * B + i] = lp[k];
    }
}

/* M-step sums of the block, added to acc (the private copy of the thread):
 *   Nk += sum_i r_ik      sx_k[a] += sum_i r_ik x_ia      sxx_k[a][b] += sum_i (r_ik x_ia) x_ib
 * Every sum over the rows i is a contiguous SIMD reduction accumulated in a register, and each
 * element of acc is read and written ONCE per block (row by row: once per row). RX is scratch
 * (D*B): RX[a*B + i] = r_ik x_ia, the same product r_ik * x_ia of the baseline. */
static void Accumulate_block(const double *restrict Xt /* in */, const double *restrict R /* in */,
                             int nb /* in */, int K /* in */, int D /* in */, int B /* in */,
                             double *restrict RX /* scratch */, double *restrict acc /* in/out */)
{
    for (int k = 0; k < K; k++)
    {
        const double *restrict rk = R + (size_t)k * B;
        double *restrict sx = acc + K + (size_t)k * D;
        double *restrict sxx = acc + K + (size_t)K * D + (size_t)k * D * D;

        double nk = 0.0;
#pragma omp simd reduction(+ : nk)
        for (int i = 0; i < nb; i++)
            nk += rk[i];
        acc[k] += nk;

        /* RX = r_k * x (per component), and its sum over the rows = sx */
        for (int a = 0; a < D; a++)
        {
            const double *restrict xa = Xt + (size_t)a * B;
            double *restrict rxa = RX + (size_t)a * B;
            double s1 = 0.0;
#pragma omp simd reduction(+ : s1)
            for (int i = 0; i < nb; i++)
            {
                const double v = rk[i] * xa[i];
                rxa[i] = v;
                s1 += v;
            }
            sx[a] += s1;
        }

        /* Lower triangle of sum_i (r_ik x_ia) x_ib: one register sum per element (a, b) */
        for (int a = 0; a < D; a++)
        {
            const double *restrict rxa = RX + (size_t)a * B;
            double *restrict row = sxx + (size_t)a * D;
            for (int b = 0; b <= a; b++)
            {
                const double *restrict xb = Xt + (size_t)b * B;
                double s2 = 0.0;
#pragma omp simd reduction(+ : s2)
                for (int i = 0; i < nb; i++)
                    s2 += rxa[i] * xb[i];
                row[b] += s2; /* ONE read + ONE write per block */
            }
        }
    }
}

/* ----------------------------------------------------------------------------------------
 * FUSED PASS, BLOCKED, on the LOCAL rows: Cholesky + E-step + local sums of the M-step, in ONE
 * parallel region (see (2)), with the rows processed in blocks of B (see (4)).
 *
 * Input:  local block ds, current parameters p and covariances w->cov (equal on all the ranks).
 * Output: w->L and w->c (cluster terms), w->acc = [Nk | sx | sxx] of the local rows (still to be
 *         summed over the ranks), w->labels, and the LOCAL log-likelihood (returned).
 *         No MPI call inside the parallel region (MPI_THREAD_FUNNELED).
 * ---------------------------------------------------------------------------------------- */
static double fused_pass(const Dataset *ds /* in */, const GMMParams *p /* in */,
                         Workspace *w /* in/out */)
{
    const int n = ds->N, D = ds->D, K = p->K, B = w->B;
    const int len = K + K * D + K * D * D;
    const int nblk = (n + B - 1) / B; /* blocks of B rows; the last one may be shorter */
    double *acc = w->acc;
    /* Local copies of the pointers: the compiler cannot prove that the stores into acc / the
     * buffers do not modify the fields of w (or the arrays they point to), so it would reload them
     * at every iteration and give up vectorizing. */
    const double *restrict X = ds->X;
    const double *restrict mu_all = p->mu;
    const double *restrict cov = w->cov;
    double *restrict L_all = w->L;
    double *restrict cterm = w->c;
    int *restrict labels = w->labels;
    double *restrict tbuf = w->tbuf;
    const size_t slice = w->slice, off_Z = w->off_Z, off_LP = w->off_LP, off_R = w->off_R;

    /* The OpenMP reduction ADDS the private copies to the value already in acc: zero it first */
    memset(acc, 0, (size_t)len * sizeof(double));

    double ll = 0.0;
    int bad = 0; /* k+1 of a cluster whose covariance cannot be factorized, 0 = all fine */
#pragma omp parallel num_threads(w->n_threads) reduction(+ : ll) reduction(+ : acc[:len])
    {
        /* My slice of the block buffers (allocated once, 64-byte aligned, no shared lines) */
        double *my = tbuf + (size_t)omp_get_thread_num() * slice;
        double *Xt = my, *Zt = my + off_Z, *LP = my + off_LP, *R = my + off_R;

        /* Cluster terms: the K Cholesky factorizations, one cluster per thread (see (2)).
         * gmm_full_factorize is called on ONE cluster (K = 1) with the pointers moved to cluster k:
         * same jitter rule, so the same factors as the serial call of the baseline. reduction(max)
         * collects a failure from any thread; the implicit barrier at the end of the "omp for"
         * makes L, c and bad visible to everybody before the main loop. */
#pragma omp for schedule(static) reduction(max : bad)
        for (int k = 0; k < K; k++)
        {
            if (gmm_full_factorize(cov + (size_t)k * D * D, p->pi + k, 1, D,
                                   L_all + (size_t)k * D * D, cterm + k) != 0)
                bad = k + 1;
        }

        /* All the threads read the same value of bad (after the barrier), so either all of them
         * enter the loop below or none does: no thread is left waiting at its implicit barrier.
         * The error is reported after the parallel region (MPI_Abort must not be called from
         * inside it). */
        if (bad == 0)
        {
            /* Main loop over the BLOCKS (every block costs the same: static). Every thread has
             * private copies of acc and ll (OpenMP reduction). */
#pragma omp for schedule(static)
            for (int blk = 0; blk < nblk; blk++)
            {
                const int i0 = blk * B;
                const int nb = (n - i0 < B) ? n - i0 : B;

                Load_block_transposed(X + (size_t)i0 * D, nb, D, B, Xt);
                /* E-step of the block: log-densities of all the rows for every cluster */
                for (int k = 0; k < K; k++)
                    Mahalanobis_block(Xt, mu_all + (size_t)k * D, L_all + (size_t)k * D * D,
                                      cterm[k], k, K, nb, D, B, Zt, LP);
                Responsibilities_block(LP, nb, K, B, R, labels + i0, &ll);
                /* M-step sums of the block, while Xt is still in the cache (Zt reused as RX) */
                Accumulate_block(Xt, R, nb, K, D, B, Zt, acc);
            }
        }
    } /* end of the parallel region: ll and acc are the sums of the private copies */

    if (bad)
        ABORT("covariance of cluster %d is not positive definite", bad - 1);
    return ll;
}

/* ----------------------------------------------------------------------------------------
 * M-step, part 2: new parameters from the GLOBAL sums in w->acc (same as the baseline)
 *
 * Every rank runs it on the same data (the result of the Allreduce), so all the ranks obtain the
 * same parameters without communicating them. Returns the number of clusters found empty.
 * ---------------------------------------------------------------------------------------- */
static int m_step_update(GMMParams *p /* in/out */, Workspace *w /* in/out */,
                         long N_total /* in */)
{
    const int K = p->K, D = p->D;
    const double *Nk = w->acc;
    const double *sx = w->acc + K;
    const double *sxx = w->acc + K + (size_t)K * D;

    int empty = 0;
    for (int k = 0; k < K; k++)
    {
        if (Nk[k] < NK_EMPTY)
        {
            /* Empty cluster: mean and covariance stay as before (dividing by ~0 would give NaN) */
            empty++;
            p->pi[k] = PI_FLOOR;
            continue;
        }
        const double inv = 1.0 / Nk[k];
        double *mu = p->mu + (size_t)k * D;
        for (int a = 0; a < D; a++)
            mu[a] = sx[(size_t)k * D + a] * inv;

        double *S = w->cov + (size_t)k * D * D;
        const double *sxxk = sxx + (size_t)k * D * D;
        for (int a = 0; a < D; a++)
            for (int b = 0; b <= a; b++)
            {
                double v = sxxk[(size_t)a * D + b] * inv - mu[a] * mu[b]; /* E[xx^T] - mu mu^T */
                if (a == b)
                    v += VAR_FLOOR; /* regularization: keeps Sigma positive definite */
                S[(size_t)a * D + b] = v;
                S[(size_t)b * D + a] = v; /* symmetric copy */
            }
        double pi = Nk[k] / (double)N_total;
        p->pi[k] = pi < PI_FLOOR ? PI_FLOOR : pi;
    }
    return empty;
}

/* ----------------------------------------------------------------------------------------
 * A complete EM run starting from the parameters p and the covariances w->cov (initialized by the
 * caller, equal on all the ranks). Every rank executes the same sequence of steps, because the
 * convergence test uses the GLOBAL log-likelihood.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    int iterations;   /* number of parameter updates performed */
    int converged;    /* 1 = tolerance reached, 0 = maximum number of iterations */
    int empty_events; /* clusters found empty (summed over the iterations) */
    double ll;        /* final GLOBAL log-likelihood, consistent with the final parameters */
    double t_estep;   /* seconds in the fused passes (this rank): Cholesky + E-step + local sums */
    double t_mstep;   /* seconds in m_step_update (this rank) */
    double t_comm;    /* seconds in the MPI_Allreduce calls (this rank) */
} EMResult;

/* Fused pass + global sum of the log-likelihood. Adds the times to res. On return w->acc holds
 * the LOCAL sums and w->labels the labels for the parameters p. */
static double pass_global(const Dataset *ds /* in */, const GMMParams *p /* in */,
                          Workspace *w /* in/out */, EMResult *res /* in/out */)
{
    double t0 = MPI_Wtime();
    double ll_local = fused_pass(ds, p, w);
    double t1 = MPI_Wtime();

    double ll; /* sum of the local log-likelihoods, available on every rank */
    MPI_Allreduce(&ll_local, &ll, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    res->t_estep += t1 - t0;
    res->t_comm += t2 - t1;
    return ll;
}

static EMResult run_em(const Dataset *ds /* in */, GMMParams *p /* in/out */,
                       Workspace *w /* in/out */, const RunArgs *a /* in */,
                       long N_total /* in */, int rank /* in */)
{
    EMResult res;
    memset(&res, 0, sizeof res);
    double ll = -INFINITY, ll_prev = -INFINITY;
    const int K = p->K;
    const int len = K + K * p->D + K * p->D * p->D;

    for (int it = 0; it < a->max_iter; it++)
    {
        ll = pass_global(ds, p, w, &res);

        if (a->verbose && rank == 0) /* only rank 0 prints, outside the timers */
            printf("  iteration %3d  log-likelihood %.10g  change %+.3e\n", it, ll,
                   isfinite(ll_prev) ? ll - ll_prev : 0.0);

        if (gmm_converged(ll, ll_prev, a->tol))
        {
            res.converged = 1;
            break; /* ll is the same on all the ranks: they all leave here. The sums of this
                    * pass are not used; w->labels are consistent with p and ll */
        }

        /* Global sums (in place), then every rank updates the parameters */
        double t0 = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, w->acc, len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        res.empty_events += m_step_update(p, w, N_total);
        double t2 = MPI_Wtime();

        res.t_comm += t1 - t0;
        res.t_mstep += t2 - t1;

        res.iterations++;
        ll_prev = ll;
    }

    /* If the loop ended because of max_iter, one more pass makes ll and the labels consistent
     * with the final parameters (as in the baseline) */
    if (!res.converged)
        ll = pass_global(ds, p, w, &res);
    res.ll = ll;
    return res;
}

/* ----------------------------------------------------------------------------------------
 * WARM-UP: throwaway EM iterations (see the baseline for the reason), with the same fused pass
 * and the same MPI calls as the real run.
 * ---------------------------------------------------------------------------------------- */
static void warmup_em(const Dataset *ds /* in */, GMMParams *p /* in/out */,
                      Workspace *w /* in/out */, int iters /* in */, long N_total /* in */)
{
    EMResult scratch; /* receives (and discards) the times */
    memset(&scratch, 0, sizeof scratch);
    const int K = p->K;
    const int len = K + K * p->D + K * p->D * p->D;
    for (int it = 0; it < iters; it++)
    {
        pass_global(ds, p, w, &scratch);
        MPI_Allreduce(MPI_IN_PLACE, w->acc, len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        m_step_update(p, w, N_total);
    }
}

/* ----------------------------------------------------------------------------------------
 * main
 *
 * Phases, in order (as in the sequential version):
 *   0. MPI initialization and reading of the arguments (every rank)
 *   1. every rank reads its block of rows
 *   2. EM (one or more initializations)
 *   3. labels gathered on rank 0 and evaluation (optional, outside the timing)
 *   4. log and summary (rank 0 only)
 * ---------------------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    /* 0. MPI. FUNNELED: only the main thread calls MPI (always outside parallel regions) */
    int provided, rank, P;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    const int n_threads = omp_get_max_threads(); /* from OMP_NUM_THREADS */

    /* Every rank parses the same command line (each needs the values). With -h or a wrong
     * option all the ranks exit together, so none is left waiting in a collective call. */
    RunArgs a;
    args_defaults(&a, VERSION_NAME);
    int rc = args_parse(argc, argv, &a);
    if (rc != 0)
    {
        MPI_Finalize();
        return rc > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    /* 1. Reading. Everybody reads the header (a few bytes) to know the number of rows; then
     *    every rank loads ITS block. If -n is given, only the first n_rows rows are used
     *    (as in the sequential version): the blocks split those rows. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    long N_file;
    int D;
    if (dataset_read_header(a.dataset, &N_file, &D) != 0)
        ABORT("cannot read the header of %s", a.dataset);
    const long N_total = a.n_rows < 0 ? N_file : a.n_rows; /* rows used, over all the ranks */
    if (N_total < 1 || N_total > N_file)
        ABORT("-n %ld out of range (the file has %ld rows)", a.n_rows, N_file);
    if (N_total < P)
        ABORT("more ranks (%d) than rows (%ld)", P, N_total);

    long first_row, n_local_l;
    block_of_rank(N_total, P, rank, &first_row, &n_local_l);

    double t_read0 = MPI_Wtime();
    Dataset ds;
    if (dataset_load(a.dataset, first_row, n_local_l, &ds) != 0)
        ABORT("rank %d: cannot load rows [%ld, %ld) of %s", rank, first_row,
              first_row + n_local_l, a.dataset);
    double t_read_local = MPI_Wtime() - t_read0;

    const int n_local = ds.N, K = a.K;
    if (K > N_total)
        ABORT("the number of clusters (%d) is larger than the number of rows (%ld)", K, N_total);

    /* Memory: parameters (replicated on every rank; the covariances are in w.cov), working space,
     * and the arrays used to choose the K initial rows and to remember the labels of the best run. */
    GMMParams p, best;
    gmm_alloc(&p, K, D);
    gmm_alloc(&best, K, D);
    /* Rows per block: -b, or the default of this version */
    const int B = a.block > 0 ? a.block : BLOCK_DEFAULT;
    Workspace w;
    workspace_alloc(&w, n_local, K, D, B, n_threads);
    uint64_t *idx = (uint64_t *)malloc((size_t)K * sizeof(uint64_t));
    double *rows = (double *)malloc((size_t)K * D * sizeof(double));
    int *best_labels = (int *)malloc((size_t)n_local * sizeof(int)); /* LOCAL rows */
    CHECK_ALLOC(idx);
    CHECK_ALLOC(rows);
    CHECK_ALLOC(best_labels);

    /* 2. EM with one or more initializations (restarts), as in the sequential version */
    double t_init_local = 0.0, t_estep = 0.0, t_mstep = 0.0, t_comm = 0.0;
    double t_warmup_local = 0.0; /* seconds of the warm-up pass on this rank, removed from t_total */
    int have_best = 0, total_iterations = 0;
    EMResult best_res;
    memset(&best_res, 0, sizeof best_res);

    /* Restart index -1 is the WARM-UP pass (only if --warmup > 0): it initializes like restart 0,
     * runs a.warmup throwaway EM iterations (warmup_em, with the same MPI calls and OpenMP
     * regions as the real run) and goes back to the top of the loop. The real restarts 0, 1, ...
     * then start again from their own initialization, so the results are the same as without
     * warm-up. The warm-up time is measured apart and subtracted from t_total; the init time of
     * the warm-up pass is not added to t_init. */
    for (int rs = (a.warmup > 0 ? -1 : 0); rs < a.restarts; rs++)
    {
        const int warm = (rs < 0);
        const double t_pass0 = MPI_Wtime(); /* start of this pass (used by the warm-up pass) */
        /* Initialization, identical to the sequential one. Every rank computes the SAME K
         * indices (same seed, no communication). The rank that owns a chosen row copies it into
         * rows[], the others leave zeros; the Allreduce(SUM) then gives every rank all the K
         * rows (adding zeros is exact). */
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        gmm_init_indices((uint64_t)N_total, K, a.seed + (uint64_t)(warm ? 0 : rs), idx);
        memset(rows, 0, (size_t)K * D * sizeof(double));
        for (int k = 0; k < K; k++)
        {
            long g = (long)idx[k]; /* global index of the chosen row */
            if (g >= first_row && g < first_row + n_local)
                memcpy(rows + (size_t)k * D, ds.X + (size_t)(g - first_row) * D,
                       (size_t)D * sizeof(double));
        }
        MPI_Allreduce(MPI_IN_PLACE, rows, K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gmm_init_from_rows(&p, rows);
        gmm_full_init_identity(w.cov, K, D); /* Sigma_k = I, as in the baseline */
        if (!warm)
            t_init_local += MPI_Wtime() - t0;

        if (warm)
        {
            /* WARM-UP: throwaway iterations on the real data and workspace (see warmup_em) */
            warmup_em(&ds, &p, &w, a.warmup, N_total);
            t_warmup_local = MPI_Wtime() - t_pass0;
            continue; /* the next iteration is restart 0; its MPI_Barrier aligns all the ranks
                       * before the real run starts */
        }

        if (a.verbose && rank == 0)
            printf("run %d (seed %llu)\n", rs, (unsigned long long)(a.seed + (uint64_t)rs));
        EMResult res = run_em(&ds, &p, &w, &a, N_total, rank);
        t_estep += res.t_estep;
        t_mstep += res.t_mstep;
        t_comm += res.t_comm;
        total_iterations += res.iterations;

        /* res.ll is the same on all the ranks, so all of them agree on which run is the best */
        if (!have_best || res.ll > best_res.ll)
        {
            have_best = 1;
            best_res = res;
            gmm_copy(&best, &p);
            memcpy(best_labels, w.labels, (size_t)n_local * sizeof(int)); /* labels of this run */
        }
    }
    double t_total_local = MPI_Wtime() - t_start - t_warmup_local; /* warm-up excluded */

    /* Times of the whole job = maximum over the ranks (valid on rank 0) */
    const double t_total = max_over_ranks(t_total_local);
    const double t_read = max_over_ranks(t_read_local);
    const double t_init = max_over_ranks(t_init_local);
    const double t_estep_max = max_over_ranks(t_estep);
    const double t_mstep_max = max_over_ranks(t_mstep);
    const double t_comm_max = max_over_ranks(t_comm);
    const double t_warmup_max = max_over_ranks(t_warmup_local);

    /* Compute time of EVERY rank (fused passes + update, no MPI calls): the spread between the
     * ranks is the load imbalance (plot P11). Valid on rank 0. */
    double *rank_compute = (rank == 0) ? (double *)malloc((size_t)P * sizeof(double)) : NULL;
    if (rank == 0)
        CHECK_ALLOC(rank_compute);
    gather_over_ranks(t_estep + t_mstep, rank_compute);

    /* 3. Evaluation (not timed). The local labels are gathered on rank 0, which loads the true
     *    labels and computes the ARI. The block sizes are known to everyone (block_of_rank), so
     *    the counts and displacements of the Gatherv are computed locally. */
    double ari = NAN, pur = NAN;
    {
        int *all_labels = NULL, *counts = NULL, *displs = NULL;
        if (rank == 0)
        {
            all_labels = (int *)malloc((size_t)N_total * sizeof(int));
            counts = (int *)malloc((size_t)P * sizeof(int));
            displs = (int *)malloc((size_t)P * sizeof(int));
            CHECK_ALLOC(all_labels);
            CHECK_ALLOC(counts);
            CHECK_ALLOC(displs);
            for (int r = 0; r < P; r++)
            {
                long f, n;
                block_of_rank(N_total, P, r, &f, &n);
                counts[r] = (int)n;
                displs[r] = (int)f;
            }
        }
        MPI_Gatherv(best_labels, n_local, MPI_INT, all_labels, counts, displs, MPI_INT, 0,
                    MPI_COMM_WORLD);
        if (rank == 0)
        {
            int32_t *truth = NULL;
            if (dataset_load_labels(a.dataset, 0, N_total, &truth) == 0)
            {
                ari = adjusted_rand_index(truth, all_labels, (int)N_total, K);
                pur = purity(truth, all_labels, (int)N_total, K);
            }
            free(truth);
        }
        free(all_labels);
        free(counts);
        free(displs);
    }

    /* 4. Log and summary: rank 0 only */
    if (rank == 0)
    {
        RunStats st;
        memset(&st, 0, sizeof st);
        st.version = a.name;
        st.dataset = a.dataset;
        st.ranks = P;
        st.threads = n_threads;
        st.N = (int)N_total;
        st.D = D;
        st.K = K;
        st.restarts = a.restarts;
        st.warmup_iters = a.warmup;
        st.t_warmup = t_warmup_max;
        st.iterations = best_res.iterations;
        st.converged = best_res.converged;
        st.loglik = best_res.ll;
        st.ari = ari;
        st.t_total = t_total;
        st.t_read = t_read;
        st.t_init = t_init;
        st.t_estep = t_estep_max;
        st.t_mstep = t_mstep_max;
        st.t_comm = t_comm_max;
        /* Analytic work for the roofline (P10): the SAME as the fused pass without blocks (blocking changes the order of the
         * operations, not their number; the block is copied from X once per pass, fused = 1). The
         * E-step runs iterations + 1 times per restart (the last one evaluates the log-likelihood). */
        st.flops = pem_flops(PEM_FULL, N_total, D, K, total_iterations + a.restarts, total_iterations);
        st.bytes = pem_bytes(1, N_total, D, K, total_iterations + a.restarts, total_iterations);
        st.rank_compute = rank_compute;
        st.n_rank_compute = P;
        st.block = B; /* written in the log ("block"), for the sweep over B */

        if (a.log_path != NULL && log_run(a.log_path, &st) != 0)
            fprintf(stderr, "warning: the log line was not written\n");

        printf("=== %s ===\n", a.name);
        printf("dataset         : %s\n", a.dataset);
        printf("covariance      : full\n");
        printf("rows x columns  : %ld x %d, K = %d\n", N_total, D, K);
        printf("ranks x threads : %d x %d\n", P, n_threads);
        printf("block           : %d rows\n", B);
        printf("seed / restarts : %llu / %d\n", (unsigned long long)a.seed, a.restarts);
        printf("warm-up         : %d iterations, %.4f s (not in the times)\n", a.warmup, t_warmup_max);
        printf("iterations      : %d (%s)\n", best_res.iterations,
               best_res.converged ? "converged" : "maximum number of iterations reached");
        printf("log-likelihood  : %.10g\n", best_res.ll);
        if (!isnan(ari) && ari > -1.5)
            printf("vs true labels  : ARI %.4f, purity %.4f\n", ari, pur);
        printf("time (s)        : total %.4f | read %.4f | init %.4f | E-step %.4f | M-step %.4f | comm %.4f\n",
               t_total, t_read, t_init, t_estep_max, t_mstep_max, t_comm_max);
        if (total_iterations > 0)
            printf("per iteration   : %.3f ms (E-step + M-step + comm)\n",
                   1e3 * (t_estep_max + t_mstep_max + t_comm_max) / total_iterations);
        if (best_res.empty_events > 0)
            printf("warning         : %d empty cluster(s) found during the iterations\n",
                   best_res.empty_events);
        if (a.log_path != NULL)
            printf("log             : %s\n", a.log_path);
    }

    /* Release of the memory */
    free(rank_compute);
    free(idx);
    free(rows);
    free(best_labels);
    workspace_free(&w);
    gmm_free(&p);
    gmm_free(&best);
    dataset_free(&ds);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
