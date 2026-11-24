/*
 * PEMC-baseline-MPI-OpenMP-full.c
 *
 * Hybrid MPI + OpenMP EM for a mixture of Gaussians with FULL covariance. It is the PARALLEL
 * baseline-MPI-OpenMP of the full model: the parallel design (row blocks, replicated parameters, OpenMP
 * loops over the local rows, MPI_Allreduce of the sums) is exactly the one of
 * PEMC-baseline-MPI-OpenMP-diagonal.c, and the model, the formulas and the numerical rules are
 * exactly those of PEMC-baseline-sequential-full.c. Read those two files first: here only what
 * changes because of the covariance is described. It must reach the same final log-likelihood,
 * the same number of iterations and the same ARI as baseline-sequential-full from the same
 * seed (up to rounding). The optimizations of the full model (PEMC-opt1/2-...-full) are
 * compared against this file, so it is deliberately the simple version: it stores the matrix
 * of responsibilities and makes two separate passes (E-step, M-step).
 *
 * WHAT CHANGES COMPARED TO THE DIAGONAL PARALLEL VERSION
 *   Parameters:  the covariances Sigma_k (K*D*D values, replicated on every rank) replace var
 *                (they are kept in the workspace, w->cov, as in baseline-sequential-full).
 *   E-step:      every rank factorizes the K matrices Sigma_k (Cholesky, K*D^3/6 operations:
 *                repeated on every rank, cheaper than communicating the factors); each row then
 *                needs a forward substitution per cluster (D^2/2 operations instead of D).
 *                The per-thread scratch buffer now holds also the vector z of the substitution.
 *                The jitter loop of the Cholesky is deterministic and the parameters are equal on
 *                all the ranks, so all the ranks take the same decisions.
 *   M-step:      the packed buffer of the Allreduce is [Nk | sx | sxx] with K + K*D + K*D*D
 *                doubles (diagonal: K + 2*K*D). sxx[a][b] = sum r x_a x_b: only the lower
 *                triangle is accumulated, but the whole block is reduced (simple, and the
 *                message size does not depend on the triangle).
 *   Communication: still exactly TWO MPI_Allreduce(SUM) per iteration, but the second one
 *                grows from O(K*D) to O(K*D^2) doubles: the communication cost is no longer
 *                negligible for large D.
 *   Memory:      the private copy of the buffer that every thread has in the OpenMP array
 *                reduction is K*D^2 doubles, instead of K*D.
 *
 * ========================================================================================
 * PARALLEL DESIGN (same as the diagonal version)
 * ========================================================================================
 *   MPI:    the N rows are split in P contiguous blocks; every rank reads only its block
 *           (dataset_load with first_row), holds a full copy of the parameters, and computes the
 *           new parameters by itself from the Allreduce results (no Bcast).
 *   OpenMP: the loop over the LOCAL rows of the E-step and of the M-step is split among the
 *           threads (ll is a reduction(+); the three sums are an array reduction). MPI is called
 *           only OUTSIDE the parallel regions (MPI_THREAD_FUNNELED is enough).
 *
 * Number of threads: environment variable OMP_NUM_THREADS (same options as
 * baseline-sequential: see args_usage()).
 *
 * NOTE ON THE RESULT: the sums are made in a different order than in baseline-sequential
 * (blocks per rank, per thread), and floating-point addition is not associative. The final
 * log-likelihood therefore equals baseline-sequential up to rounding, not bit by bit.
 *
 * ========================================================================================
 * TIMES (reported in the log, maximum over the ranks)
 * ========================================================================================
 *   t_estep, t_mstep   computation only (the loops, without the MPI calls)
 *   t_comm             the MPI_Allreduce calls of the iterations (includes the waiting for the
 *                      slowest rank, i.e. the load imbalance)
 *   Barrier only before the global timers (start of read / init / EM).
 *
 * Usage: see args_usage() (or run with -h).  Example:
 *   OMP_NUM_THREADS=4 mpiexec -n 2 bin/PEMC-baseline-MPI-OpenMP-full -d FILE.bin -k 4
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
#define VERSION_NAME "baseline-MPI-OpenMP-full"

/* ----------------------------------------------------------------------------------------
 * Working memory of EM (of ONE rank)
 *
 * All the arrays are allocated ONCE before starting and reused at every iteration.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    double *r;       /* n_local*K: responsibilities of the LOCAL rows, r[i*K + k] */
    double *cov;     /* K*D*D: covariance matrices Sigma_k (full, symmetric, row-major) */
    double *L;       /* K*D*D: Cholesky factors of Sigma_k (lower triangle used), rebuilt every E-step */
    double *c;       /* K: part of the log-density that depends only on the cluster */
    double *logp;    /* n_threads*stride: log(pi_k N(x_i | ...)), one slice per thread
                        (a buffer shared by the threads would be a data race) */
    double *z;       /* n_threads*stride: scratch vector of the forward substitution, one slice
                        per thread */
    int stride;      /* distance (in doubles) between the slices of two threads: max(K, D) rounded
                        up to a multiple of 8 doubles = 64 bytes. malloc aligns only to 16 bytes, so
                        a slice can still share a cache line with the slice of its neighbour (false
                        sharing): removed in opt1, see its header (3) */
    double *acc;     /* K + K*D + K*D*D: the M-step sums, in one contiguous buffer so that a single
                        MPI_Allreduce is enough. Layout:
                          acc[0 .. K-1]              Nk  = sum_i r_ik
                          acc[K .. K+K*D-1]          sx  = sum_i r_ik x_id        (index k*D + d)
                          acc[K+K*D .. end]          sxx = sum_i r_ik x_ia x_ib   (index k*D*D + a*D + b,
                                                     lower triangle b <= a only) */
} Workspace;

static void workspace_alloc(Workspace *w, int n_local, int K, int D, int n_threads)
{
    const int m = K > D ? K : D;
    w->stride = (m + 7) / 8 * 8;
    w->r = (double *)malloc((size_t)n_local * K * sizeof(double));
    w->cov = (double *)malloc((size_t)K * D * D * sizeof(double));
    w->L = (double *)malloc((size_t)K * D * D * sizeof(double));
    w->c = (double *)malloc((size_t)K * sizeof(double));
    w->logp = (double *)malloc((size_t)n_threads * w->stride * sizeof(double));
    w->z = (double *)malloc((size_t)n_threads * w->stride * sizeof(double));
    w->acc = (double *)malloc((size_t)(K + K * D + (size_t)K * D * D) * sizeof(double));
    CHECK_ALLOC(w->r);
    CHECK_ALLOC(w->cov);
    CHECK_ALLOC(w->L);
    CHECK_ALLOC(w->c);
    CHECK_ALLOC(w->logp);
    CHECK_ALLOC(w->z);
    CHECK_ALLOC(w->acc);
}

static void workspace_free(Workspace *w)
{
    free(w->r);
    free(w->cov);
    free(w->L);
    free(w->c);
    free(w->logp);
    free(w->z);
    free(w->acc);
}

/* ----------------------------------------------------------------------------------------
 * E-step on the LOCAL rows: responsibilities and local log-likelihood
 *
 * Input:  local block of data ds and current parameters p (equal on all the ranks).
 * Output: fills w->r (local rows) and returns the LOCAL log-likelihood (the caller sums it
 *         over the ranks). No MPI call here: it is pure computation.
 * ---------------------------------------------------------------------------------------- */
static double e_step(const Dataset *ds /* in */, const GMMParams *p /* in */,
                     Workspace *w /* in/out */)
{
    const int n = ds->N, D = ds->D, K = p->K;

    /* Part 1: terms that depend only on the cluster (K*D^3/6 operations: every rank repeats them,
     * it is cheaper than communicating them): Cholesky factors and c[k] (see gmm.h, full covariance section). The
     * matrices are equal on all the ranks, so every rank takes the same decisions. */
    int bad = gmm_full_factorize(w->cov, p->pi, K, D, w->L, w->c);
    if (bad)
        ABORT("covariance of cluster %d is not positive definite", bad - 1);

    /* Part 2: the local rows are split among the threads (static: every row costs the same, so
     * equal chunks are balanced and have no scheduling overhead). Every thread works on
     * different rows, so r[i*K ..] is written without conflicts; ll is a reduction: every thread
     * sums in a private variable, OpenMP adds them at the end. */
    double ll = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : ll)
    for (int i = 0; i < n; i++)
    {
        const double *x = ds->X + (size_t)i * D; /* beginning of row i */
        const int tid = omp_get_thread_num();
        double *logp = w->logp + (size_t)tid * w->stride; /* my slices */
        double *z = w->z + (size_t)tid * w->stride;

        for (int k = 0; k < K; k++)
        {
            const double *mu = p->mu + (size_t)k * D;
            const double *L = w->L + (size_t)k * D * D;
            /* forward substitution L z = x - mu; s = |z|^2 = squared Mahalanobis distance */
            double s = 0.0;
            for (int a = 0; a < D; a++)
            {
                double t = x[a] - mu[a];
                const double *La = L + (size_t)a * D;
                for (int b = 0; b < a; b++)
                    t -= La[b] * z[b];
                t /= La[a];
                z[a] = t;
                s += t * t;
            }
            logp[k] = w->c[k] - 0.5 * s; /* log(pi_k N(x_i | mu_k, Sigma_k)) */
        }

        /* Normalization with log-sum-exp (see baseline-sequential for why) */
        double lse = log_sum_exp(logp, K);
        ll += lse;
        double *ri = w->r + (size_t)i * K;
        for (int k = 0; k < K; k++)
            ri[k] = exp(logp[k] - lse);
    }
    return ll;
}

/* ----------------------------------------------------------------------------------------
 * M-step, part 1: the three sums over the LOCAL rows, written in w->acc
 *
 * Input:  local data ds and local responsibilities w->r.
 * Output: w->acc = [Nk | sx | sxx] of the local rows only (still to be summed over the ranks).
 * ---------------------------------------------------------------------------------------- */
static void m_step_local_sums(const Dataset *ds /* in */, int K /* in */,
                              Workspace *w /* in/out */)
{
    const int n = ds->N, D = ds->D;
    const int len = K + K * D + K * D * D; /* total length of the buffer */
    double *acc = w->acc;

    /* The buffer still holds the previous iteration: it is zeroed (the OpenMP reduction
     * ADDS the private copies of the threads to the value already in acc). */
    memset(acc, 0, (size_t)len * sizeof(double));

    /* Array reduction: every thread gets a private copy of acc (initialized to 0), accumulates
     * its rows in it with no synchronization, and at the end of the loop the copies are added to
     * acc. The cost is a private buffer of len doubles per thread (K + K*D + K*D^2). */
#pragma omp parallel for schedule(static) reduction(+ : acc[:len])
    for (int i = 0; i < n; i++)
    {
        const double *x = ds->X + (size_t)i * D;
        const double *ri = w->r + (size_t)i * K;
        for (int k = 0; k < K; k++)
        {
            double rik = ri[k];
            acc[k] += rik;                        /* Nk  */
            double *sx = acc + K + (size_t)k * D; /* sx  of cluster k */
            double *sxx = acc + K + (size_t)K * D + (size_t)k * D * D; /* sxx of cluster k */
            for (int a = 0; a < D; a++)
            {
                double rx = rik * x[a];
                sx[a] += rx;
                double *row = sxx + (size_t)a * D;
                for (int b = 0; b <= a; b++) /* lower triangle only: sxx is symmetric */
                    row[b] += rx * x[b];
            }
        }
    }
}

/* ----------------------------------------------------------------------------------------
 * M-step, part 2: new parameters from the GLOBAL sums in w->acc
 *
 * Every rank runs it on the same data (the result of the Allreduce), so all the ranks obtain
 * the same parameters without communicating them. N_total is the number of rows of ALL the
 * ranks. Returns the number of clusters found empty. Same rules as baseline-sequential-full
 * (VAR_FLOOR added to the diagonal of Sigma_k).
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
 * A complete EM run starting from the parameters p (initialized by the caller, equal on all
 * the ranks). Every rank executes the same sequence of steps and the same number of
 * iterations, because the convergence test uses the GLOBAL log-likelihood.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    int iterations;   /* number of M-steps performed */
    int converged;    /* 1 = tolerance reached, 0 = maximum number of iterations */
    int empty_events; /* clusters found empty (summed over the iterations) */
    double ll;        /* final GLOBAL log-likelihood, consistent with the final parameters */
    double t_estep;   /* seconds in the E-step loops (this rank) */
    double t_mstep;   /* seconds in the M-step loops (this rank) */
    double t_comm;    /* seconds in the MPI_Allreduce calls (this rank) */
} EMResult;

/* E-step + global sum of the log-likelihood. Adds the times to res. */
static double e_step_global(const Dataset *ds /* in */, const GMMParams *p /* in */,
                            Workspace *w /* in/out */, EMResult *res /* in/out */)
{
    double t0 = MPI_Wtime();
    double ll_local = e_step(ds, p, w);
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

    /* Same structure as baseline-sequential: E-step, convergence test, M-step */
    for (int it = 0; it < a->max_iter; it++)
    {
        ll = e_step_global(ds, p, w, &res);

        if (a->verbose && rank == 0) /* only rank 0 prints, outside the timers */
            printf("  iteration %3d  log-likelihood %.10g  change %+.3e\n", it, ll,
                   isfinite(ll_prev) ? ll - ll_prev : 0.0);

        if (gmm_converged(ll, ll_prev, a->tol))
        {
            res.converged = 1;
            break; /* ll is the same on all the ranks: they all leave here */
        }

        /* M-step: local sums, then ONE Allreduce on the packed buffer (in place: the same
         * buffer is input and output), then every rank updates the parameters */
        double t0 = MPI_Wtime();
        m_step_local_sums(ds, K, w);
        double t1 = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, w->acc, len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        res.empty_events += m_step_update(p, w, N_total);
        double t3 = MPI_Wtime();

        res.t_mstep += (t1 - t0) + (t3 - t2);
        res.t_comm += t2 - t1;

        res.iterations++;
        ll_prev = ll;
    }

    /* If the loop ended because of max_iter, one more E-step makes ll and the responsibilities
     * consistent with the final parameters (as in baseline-sequential) */
    if (!res.converged)
        ll = e_step_global(ds, p, w, &res);
    res.ll = ll;
    return res;
}

/* ----------------------------------------------------------------------------------------
 * WARM-UP
 *
 * Runs 'iters' complete EM iterations (E-step + M-step) that are thrown away: no convergence
 * test, no result kept, not counted in any time. Their only purpose is to bring the machine to
 * the state it has during the real run, so the measured times are not polluted by one-time costs:
 *   - first touch of the memory (page faults of the workspace and of the dataset)
 *   - creation of the OpenMP thread pool and first parallel regions
 *   - CPU frequency ramp-up and warm caches / TLB / branch predictors
 * In the MPI versions it also runs the same MPI_Allreduce calls as the real run: the first
 * Allreduce is much slower than the following ones (connections are opened lazily).
 * It is called from main() in the warm-up pass (restart index -1), right after the
 * initialization; the parameters it modifies are then re-initialized with the SAME seed of
 * restart 0, so the final result is identical with and without warm-up.
 * ---------------------------------------------------------------------------------------- */
static void warmup_em(const Dataset *ds, GMMParams *p, Workspace *w, int iters, long N_total)
{
    EMResult scratch; /* receives (and discards) the times that e_step_global would add */
    memset(&scratch, 0, sizeof scratch);
    const int K = p->K;
    const int len = K + K * p->D + K * p->D * p->D;
    for (int it = 0; it < iters; it++)
    {
        e_step_global(ds, p, w, &scratch);
        m_step_local_sums(ds, K, w);
        MPI_Allreduce(MPI_IN_PLACE, w->acc, len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        m_step_update(p, w, N_total);
    }
}

/* ----------------------------------------------------------------------------------------
 * main
 *
 * Phases, in order (as in baseline-sequential):
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
     *    (as in baseline-sequential): the blocks split those rows. */
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

    /* Memory: parameters (replicated on every rank; the covariances are in w.cov), working space, and the arrays used to
     * choose the K initial rows and to remember the labels of the best run. */
    GMMParams p, best;
    gmm_alloc(&p, K, D);
    gmm_alloc(&best, K, D);
    Workspace w;
    workspace_alloc(&w, n_local, K, D, n_threads);
    uint64_t *idx = (uint64_t *)malloc((size_t)K * sizeof(uint64_t));
    double *rows = (double *)malloc((size_t)K * D * sizeof(double));
    int *best_labels = (int *)malloc((size_t)n_local * sizeof(int)); /* LOCAL rows */
    CHECK_ALLOC(idx);
    CHECK_ALLOC(rows);
    CHECK_ALLOC(best_labels);

    /* 2. EM with one or more initializations (restarts), as in baseline-sequential */
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
        /* Initialization, identical to baseline-sequential. Every rank computes the SAME K
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
        gmm_full_init_identity(w.cov, K, D); /* Sigma_k = I, as var = 1 in the diagonal version */
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
            gmm_assign(w.r, n_local, K, best_labels); /* w.r refers to this run */
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

    /* Compute time of EVERY rank (E-step + M-step loops, no MPI calls): the spread between the
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
        /* Analytic work for the roofline (P10). The E-step runs iterations + 1 times per restart
         * (the last one evaluates the log-likelihood). fused = 0: baseline-MPI-OpenMP makes two passes
         * over the data and stores the responsibilities. An optimization that fuses the passes
         * must pass fused = 1 here. */
        st.flops = pem_flops(PEM_FULL, N_total, D, K, total_iterations + a.restarts, total_iterations);
        st.bytes = pem_bytes(0, N_total, D, K, total_iterations + a.restarts, total_iterations);
        st.rank_compute = rank_compute;
        st.n_rank_compute = P;

        if (a.log_path != NULL && log_run(a.log_path, &st) != 0)
            fprintf(stderr, "warning: the log line was not written\n");

        printf("=== %s ===\n", a.name);
        printf("dataset         : %s\n", a.dataset);
        printf("covariance      : full\n");
        printf("rows x columns  : %ld x %d, K = %d\n", N_total, D, K);
        printf("ranks x threads : %d x %d\n", P, n_threads);
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
