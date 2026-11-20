/*
 * PEMC-opt2-MPI-OpenMP-diagonal.c
 *
 * Hybrid MPI + OpenMP EM for a mixture of Gaussians with DIAGONAL covariance.
 * OPTIMIZATION 2: COMMUNICATION. It starts from PEMC-baseline-MPI-OpenMP-diagonal.c and changes
 * ONLY how the ranks communicate: the computation (E-step and M-step loops, OpenMP parallelization,
 * workspace, stored responsibilities, two passes over the data) is the one of the baseline, line
 * by line. So "opt2 vs baseline" measures the effect of the communication alone (opt1 is a
 * separate optimization of the computation and is not included here).
 * It must reach the same final log-likelihood, the same number of iterations and the same ARI as
 * the baseline starting from the same seed.
 *
 * ========================================================================================
 * WHAT CHANGES WITH RESPECT TO THE BASELINE
 * ========================================================================================
 * Baseline, per iteration: TWO MPI_Allreduce(SUM), i.e. two synchronization points.
 *     E-step (local) -> Allreduce of 1 double (log-likelihood) -> convergence test
 *     M-step (local) -> Allreduce of K + 2*K*D doubles (Nk, sx, sx2)       -> new parameters
 *
 * Opt2, per iteration: ONE MPI_Allreduce(SUM).
 *     E-step (local) -> M-step local sums -> ONE Allreduce of K + 2*K*D + 1 doubles
 *                       [Nk | sx | sx2 | ll]   -> convergence test -> new parameters
 *   The local log-likelihood travels in the last slot of the same buffer as the M-step sums.
 *
 * Why it is faster:
 *   (1) LATENCY. The collective that disappears is the one of the log-likelihood (1 double): its
 *       cost is only latency (log2(P) steps over the network between nodes), and appending that
 *       double to the sums adds almost nothing. On most datasets the sums are small too
 *       (K + 2*K*D doubles: from a few hundred bytes to a few KB), so both collectives are
 *       latency-bound and one instead of two halves the cost. Exception: gene_rnaseq (K = 5,
 *       D = 20 264: ~1.6 MB), where the bandwidth dominates and the relative saving is small.
 *       Cost model: Thakur et al. 2005 (doc/papers_and_references.md #T).
 *   (2) WAITING. Every Allreduce is a synchronization point: a rank that finishes early waits for
 *       the slowest one. The baseline waits twice per iteration (after the E-step and after the
 *       M-step), opt2 once, after both: the waits of the two phases are not added any more.
 *   The gain is expected where the communication matters: several nodes, many ranks, small
 *   datasets (short iterations). With one rank (or one node and big data) it should be ~1.0x.
 *
 * Why the result does not change:
 *   - the global log-likelihood of iteration t is computed from the SAME parameters as in the
 *     baseline (the ones before the M-step of iteration t), so the convergence test makes the
 *     same decision at the same iteration;
 *   - the M-step sums are the same numbers reduced in the same buffer layout, so the new
 *     parameters are the same;
 *   - when the test says "converged", the M-step sums already computed in that iteration are
 *     simply NOT applied: the parameters stay those of the last E-step, as in the baseline.
 *   Cost of this: on the LAST iteration of a converged run the local M-step sums are computed and
 *   then thrown away (one extra M-step loop per run, out of tens or hundreds of iterations).
 *   That is the price of sending the log-likelihood together with the sums.
 *
 * Everything else (MPI only outside the parallel regions, MPI_THREAD_FUNNELED, rows split in
 * contiguous blocks, parameters replicated, no Bcast) is as in the baseline: read the baseline
 * header for the parallel design.
 *
 * ========================================================================================
 * TIMES (reported in the log, maximum over the ranks)
 * ========================================================================================
 *   t_estep, t_mstep   computation only (the loops, without the MPI calls). t_mstep includes the
 *                      extra M-step loop of the converged iteration (it is real work of opt2)
 *   t_comm             the MPI_Allreduce calls of the iterations, including the waiting for the
 *                      slowest rank (load imbalance), as in the baseline
 *   Barrier only before the global timers (start of read / init / EM), as in the baseline.
 *
 * Usage: see args_usage() (or run with -h).  Example:
 *   OMP_NUM_THREADS=4 mpiexec -n 2 bin/PEMC-opt2-MPI-OpenMP-diagonal -d FILE.bin -k 4
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
#define VERSION_NAME "opt2-MPI-OpenMP-diagonal"

/* ----------------------------------------------------------------------------------------
 * Working memory of EM (of ONE rank)
 *
 * All the arrays are allocated ONCE before starting and reused at every iteration.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    double *r;       /* n_local*K: responsibilities of the LOCAL rows, r[i*K + k] */
    double *c;       /* K: part of the log-density that depends only on the cluster */
    double *inv_var; /* K*D: 1 / var, computed once per iteration */
    double *logp;    /* n_threads*logp_stride: log(pi_k N(x_i | ...)), one slice per thread
                        (a buffer shared by the threads would be a data race) */
    int logp_stride; /* distance (in doubles) between the slices of two threads: K rounded up to
                        a multiple of 8 doubles = 64 bytes. malloc aligns only to 16 bytes, so a
                        slice can still share a cache line with the slice of its neighbour (false
                        sharing): kept as in the baseline on purpose, opt2 changes only the
                        communication */
    double *acc;     /* K + 2*K*D + 1: the M-step sums AND the log-likelihood, in one contiguous
                        buffer so that a single MPI_Allreduce per iteration is enough (OPT2).
                        Layout:
                          acc[0 .. K-1]            Nk  = sum_i r_ik
                          acc[K .. K+K*D-1]        sx  = sum_i r_ik x_id     (index k*D + d)
                          acc[K+K*D .. K+2*K*D-1]  sx2 = sum_i r_ik x_id^2   (index k*D + d)
                          acc[K+2*K*D]             ll  = sum_i log p(x_i)    (OPT2: extra slot) */
} Workspace;

static void workspace_alloc(Workspace *w, int n_local, int K, int D, int n_threads)
{
    w->logp_stride = (K + 7) / 8 * 8;
    w->r = (double *)malloc((size_t)n_local * K * sizeof(double));
    w->c = (double *)malloc((size_t)K * sizeof(double));
    w->inv_var = (double *)malloc((size_t)K * D * sizeof(double));
    w->logp = (double *)malloc((size_t)n_threads * w->logp_stride * sizeof(double));
    w->acc = (double *)malloc((size_t)(K + 2 * K * D + 1) * sizeof(double)); /* +1: ll (OPT2) */
    CHECK_ALLOC(w->r);
    CHECK_ALLOC(w->c);
    CHECK_ALLOC(w->inv_var);
    CHECK_ALLOC(w->logp);
    CHECK_ALLOC(w->acc);
}

static void workspace_free(Workspace *w)
{
    free(w->r);
    free(w->c);
    free(w->inv_var);
    free(w->logp);
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

    /* Part 1: terms that depend only on the cluster (K*D operations: every rank repeats them,
     * it is cheaper than communicating them). See the sequential version for the formula. */
    for (int k = 0; k < K; k++)
    {
        double sum_log_var = 0.0;
        for (int d = 0; d < D; d++)
        {
            double v = p->var[(size_t)k * D + d];
            w->inv_var[(size_t)k * D + d] = 1.0 / v;
            sum_log_var += log(v);
        }
        w->c[k] = log(p->pi[k]) - 0.5 * (D * LOG_2PI + sum_log_var);
    }

    /* Part 2: the local rows are split among the threads (static: every row costs the same, so
     * equal chunks are balanced and have no scheduling overhead). Every thread works on
     * different rows, so r[i*K ..] is written without conflicts; ll is a reduction: every thread
     * sums in a private variable, OpenMP adds them at the end. */
    double ll = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : ll)
    for (int i = 0; i < n; i++)
    {
        const double *x = ds->X + (size_t)i * D; /* beginning of row i */
        double *logp = w->logp + (size_t)omp_get_thread_num() * w->logp_stride; /* my slice */

        for (int k = 0; k < K; k++)
        {
            const double *mu = p->mu + (size_t)k * D;
            const double *iv = w->inv_var + (size_t)k * D;
            double s = 0.0; /* squared Mahalanobis distance of the row from the cluster */
            for (int d = 0; d < D; d++)
            {
                double diff = x[d] - mu[d];
                s += diff * diff * iv[d];
            }
            logp[k] = w->c[k] - 0.5 * s; /* log(pi_k N(x_i | mu_k, var_k)) */
        }

        /* Normalization with log-sum-exp (see the sequential version for why) */
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
 * Output: w->acc = [Nk | sx | sx2] of the local rows only (still to be summed over the ranks).
 *         The last slot (ll) is not touched here: the caller writes it (OPT2).
 * ---------------------------------------------------------------------------------------- */
static void m_step_local_sums(const Dataset *ds /* in */, int K /* in */,
                              Workspace *w /* in/out */)
{
    const int n = ds->N, D = ds->D;
    const int len = K + 2 * K * D; /* total length of the buffer */
    double *acc = w->acc;

    /* The buffer still holds the previous iteration: it is zeroed (the OpenMP reduction
     * ADDS the private copies of the threads to the value already in acc). */
    memset(acc, 0, (size_t)len * sizeof(double));

    /* Array reduction: every thread gets a private copy of acc (initialized to 0), accumulates
     * its rows in it with no synchronization, and at the end of the loop the copies are added to
     * acc. The cost is a private buffer of len doubles per thread (K + 2*K*D: small). */
#pragma omp parallel for schedule(static) reduction(+ : acc[:len])
    for (int i = 0; i < n; i++)
    {
        const double *x = ds->X + (size_t)i * D;
        const double *ri = w->r + (size_t)i * K;
        for (int k = 0; k < K; k++)
        {
            double rik = ri[k];
            acc[k] += rik;                           /* Nk  */
            double *sx = acc + K + (size_t)k * D;    /* sx  of cluster k */
            double *sx2 = sx + (size_t)K * D;        /* sx2 of cluster k */
            for (int d = 0; d < D; d++)
            {
                double rx = rik * x[d];
                sx[d] += rx;
                sx2[d] += rx * x[d];
            }
        }
    }
}

/* ----------------------------------------------------------------------------------------
 * M-step, part 2: new parameters from the GLOBAL sums in w->acc
 *
 * Every rank runs it on the same data (the result of the Allreduce), so all the ranks obtain
 * the same parameters without communicating them. N_total is the number of rows of ALL the
 * ranks. Returns the number of clusters found empty. Same rules as the sequential version.
 * ---------------------------------------------------------------------------------------- */
static int m_step_update(GMMParams *p /* in/out */, const Workspace *w /* in */,
                         long N_total /* in */)
{
    const int K = p->K, D = p->D;
    const double *Nk = w->acc;
    const double *sx = w->acc + K;
    const double *sx2 = w->acc + K + (size_t)K * D;

    int empty = 0;
    for (int k = 0; k < K; k++)
    {
        if (Nk[k] < NK_EMPTY)
        {
            /* Empty cluster: mean and variances stay as before (dividing by ~0 would give NaN) */
            empty++;
            p->pi[k] = PI_FLOOR;
            continue;
        }
        for (int d = 0; d < D; d++)
        {
            double mean = sx[(size_t)k * D + d] / Nk[k];
            double var = sx2[(size_t)k * D + d] / Nk[k] - mean * mean; /* E[x^2] - E[x]^2 */
            if (var < VAR_FLOOR)
                var = VAR_FLOOR;
            p->mu[(size_t)k * D + d] = mean;
            p->var[(size_t)k * D + d] = var;
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
    int m_passes;     /* M-step loops executed (OPT2: iterations + 1 if converged), for the
                         analytic work of the roofline */
} EMResult;

/* E-step + global sum of the log-likelihood (1 double). OPT2: used ONLY after the loop, when the
 * run stops at max_iter (one call per run, as in the baseline); inside the loop the log-likelihood
 * travels with the M-step sums (see iteration_global). Adds the times to res. */
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

/* OPT2: one iteration with ONE collective call.
 * E-step and local M-step sums (the same two loops of the baseline), then the local log-likelihood
 * is written in the last slot of w->acc and the WHOLE buffer [Nk | sx | sx2 | ll] is summed over
 * the ranks with a single MPI_Allreduce. Returns the GLOBAL log-likelihood of the CURRENT
 * parameters (the same value the baseline gets from its first Allreduce); w->acc then holds the
 * global M-step sums, ready for m_step_update. Adds the times to res. */
static double iteration_global(const Dataset *ds /* in */, const GMMParams *p /* in */,
                               Workspace *w /* in/out */, EMResult *res /* in/out */)
{
    const int K = p->K;
    const int len = K + 2 * K * p->D; /* M-step sums; the ll slot is acc[len] */

    double t0 = MPI_Wtime();
    double ll_local = e_step(ds, p, w);
    double t1 = MPI_Wtime();
    m_step_local_sums(ds, K, w);
    w->acc[len] = ll_local;
    double t2 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, w->acc, len + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    res->t_estep += t1 - t0;
    res->t_mstep += t2 - t1;
    res->t_comm += t3 - t2;
    res->m_passes++;
    return w->acc[len];
}

static EMResult run_em(const Dataset *ds /* in */, GMMParams *p /* in/out */,
                       Workspace *w /* in/out */, const RunArgs *a /* in */,
                       long N_total /* in */, int rank /* in */)
{
    EMResult res;
    memset(&res, 0, sizeof res);
    double ll = -INFINITY, ll_prev = -INFINITY;

    /* Same decisions as the baseline (E-step, convergence test, M-step), but the log-likelihood
     * and the M-step sums arrive together from ONE Allreduce (OPT2) */
    for (int it = 0; it < a->max_iter; it++)
    {
        ll = iteration_global(ds, p, w, &res);

        if (a->verbose && rank == 0) /* only rank 0 prints, outside the timers */
            printf("  iteration %3d  log-likelihood %.10g  change %+.3e\n", it, ll,
                   isfinite(ll_prev) ? ll - ll_prev : 0.0);

        if (gmm_converged(ll, ll_prev, a->tol))
        {
            res.converged = 1;
            break; /* ll is the same on all the ranks: they all leave here. The M-step sums of
                    * this iteration are NOT applied: the parameters stay those of the E-step,
                    * as in the baseline */
        }

        /* M-step: the global sums are already in w->acc, every rank updates the parameters */
        double t0 = MPI_Wtime();
        res.empty_events += m_step_update(p, w, N_total);
        res.t_mstep += MPI_Wtime() - t0;

        res.iterations++;
        ll_prev = ll;
    }

    /* If the loop ended because of max_iter, one more E-step makes ll and the responsibilities
     * consistent with the final parameters (as in the sequential version) */
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
 * In the MPI versions it also runs the same MPI_Allreduce calls as the real run (OPT2: the single
 * fused one): the first Allreduce is much slower than the following ones (connections are
 * opened lazily).
 * It is called from main() in the warm-up pass (restart index -1), right after the
 * initialization; the parameters it modifies are then re-initialized with the SAME seed of
 * restart 0, so the final result is identical with and without warm-up.
 * ---------------------------------------------------------------------------------------- */
static void warmup_em(const Dataset *ds, GMMParams *p, Workspace *w, int iters, long N_total)
{
    EMResult scratch; /* receives (and discards) the times that iteration_global adds */
    memset(&scratch, 0, sizeof scratch);
    for (int it = 0; it < iters; it++)
    {
        iteration_global(ds, p, w, &scratch);
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

    /* Memory: parameters (replicated on every rank), working space, and the arrays used to
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

    /* 2. EM with one or more initializations (restarts), as in the sequential version */
    double t_init_local = 0.0, t_estep = 0.0, t_mstep = 0.0, t_comm = 0.0;
    double t_warmup_local = 0.0; /* seconds of the warm-up pass on this rank, removed from t_total */
    int have_best = 0, total_iterations = 0;
    long total_mpasses = 0; /* M-step loops executed (OPT2: one more per converged run) */
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
        total_mpasses += res.m_passes;

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
         * (the last one evaluates the log-likelihood). fused = 0: opt2 keeps the two passes of the
         * baseline and the stored responsibilities. OPT2: the M-step loop runs total_mpasses times
         * (one extra time in every converged run, see the header). */
        st.flops = pem_flops(PEM_DIAGONAL, N_total, D, K, total_iterations + a.restarts, total_mpasses);
        st.bytes = pem_bytes(0, N_total, D, K, total_iterations + a.restarts, total_mpasses);
        st.rank_compute = rank_compute;
        st.n_rank_compute = P;

        if (a.log_path != NULL && log_run(a.log_path, &st) != 0)
            fprintf(stderr, "warning: the log line was not written\n");

        printf("=== %s ===\n", a.name);
        printf("dataset         : %s\n", a.dataset);
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
