/*
 * PEMC-opt1-MPI-OpenMP-diagonal.c
 *
 * OPTIMIZATION 1 of the hybrid MPI + OpenMP EM, DIAGONAL covariance. Same model, same formulas and
 * same results as PEMC-baseline-MPI-OpenMP-diagonal.c (read it first); what changes is HOW the
 * work is organized: algorithm and memory access, not communication (that is opt2).
 *
 * ========================================================================================
 * WHAT CHANGES WITH RESPECT TO THE BASELINE
 * ========================================================================================
 * (1) FUSION OF E-STEP AND M-STEP
 *     baseline: E-step loop  : reads X, writes the responsibilities r (n_local*K doubles)
 *               M-step loop  : reads X and r again, accumulates Nk, sum r*x, sum r*x^2
 *     opt1    : ONE loop over the rows. For each row the K responsibilities are computed in a tiny
 *               per-thread buffer (K doubles, stays in the L1 cache) and used immediately to update
 *               Nk, sx, sx2. The matrix r does not exist any more, and X is read ONCE per
 *               iteration instead of twice: fewer bytes moved from the main memory
 *               (pem_bytes(fused = 1)), higher arithmetic intensity (roofline).
 *     The labels (argmax of the responsibilities of the row) are stored during the same pass,
 *     so the final clustering is available without r.
 *
 * (2) ONE PARALLEL REGION PER ITERATION
 *     baseline: two "parallel for" per iteration (E-step, M-step), each with its own fork/join,
 *               and the cluster terms (inv_var, c) computed by the master thread alone.
 *     opt1    : a single "#pragma omp parallel" that contains: the cluster terms (split among the
 *               threads), the loop over the rows (E+M fused) and the reductions of ll and of the
 *               sums. One fork/join and one creation of the private copies per iteration.
 *
 * (3) NO FALSE SHARING ON THE PER-THREAD BUFFER
 *     baseline: the K values of the row live in ONE malloc'ed array w->logp, one slice per thread
 *               (stride K rounded to 8 doubles = 64 bytes). malloc aligns only to 16 bytes, so the
 *               slice of a thread shares a 64-byte cache line with the slice of its neighbour:
 *               every write of one thread invalidates the line in the cache of the other
 *               (false sharing), at every row. The fused loop uses the buffer much more (written,
 *               overwritten with r_ik, read K*D times by the M-step part), so with this layout
 *               opt1 measured on the cluster was ~2x SLOWER than the baseline with 2-4 threads
 *               per rank (and as fast or faster with 1 thread per rank, where no sharing exists).
 *     opt1    : the buffer is a local array of the thread (double buf[K], declared inside the
 *               parallel region): it lives on the stack of that thread, no line is shared.
 *               Measured with an OpenMP replica of the loop, 8 threads: 1.7-2x faster.
 *
 * Communication is the same as the baseline on purpose (two MPI_Allreduce per iteration: ll, then
 * the packed sums), so the effect of (1)-(3) is measured alone.
 *
 * ========================================================================================
 * CONVERGENCE AND THE SUMS OF THE LAST ITERATION
 * ========================================================================================
 * ll must be known (summed over the ranks) BEFORE deciding whether to do the M-step. With the
 * fusion the local sums are already computed by then. The order is therefore:
 *     fused pass -> Allreduce(ll) -> converged?  yes: stop (sums discarded, no second Allreduce)
 *                                                no : Allreduce(sums) -> update parameters
 * The only extra work with respect to the baseline is the accumulation of the sums in the last
 * pass (thrown away): one pass out of all those performed. The number of iterations and the final
 * log-likelihood are the same as the baseline.
 *
 * TIMES: the fused pass cannot be split in E and M, so it is reported in t_estep; t_mstep is only
 * the update of the parameters from the global sums (K*D operations); t_comm as in the baseline.
 *
 * Usage: see args_usage() (or run with -h).  Example:
 *   OMP_NUM_THREADS=4 mpiexec -n 2 bin/PEMC-opt1-MPI-OpenMP-diagonal -d FILE.bin -k 4
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
#define VERSION_NAME "opt1-MPI-OpenMP-diagonal"

/* ----------------------------------------------------------------------------------------
 * Working memory of EM (of ONE rank). Allocated ONCE, reused at every iteration.
 * There is NO matrix of responsibilities: only the labels of the last pass.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    int *labels;     /* n_local: argmax_k r_ik of the LAST fused pass */
    double *c;       /* K: part of the log-density that depends only on the cluster */
    double *inv_var; /* K*D: 1 / var, computed once per iteration */
    double *acc;     /* K + 2*K*D: Nk | sx | sx2, contiguous (one MPI_Allreduce), as in the baseline */
    /* No per-thread buffer here: every thread keeps its K values in a local array (see (3)) */
} Workspace;

static void workspace_alloc(Workspace *w, int n_local, int K, int D)
{
    w->labels = (int *)malloc((size_t)n_local * sizeof(int));
    w->c = (double *)malloc((size_t)K * sizeof(double));
    w->inv_var = (double *)malloc((size_t)K * D * sizeof(double));
    w->acc = (double *)malloc((size_t)(K + 2 * K * D) * sizeof(double));
    CHECK_ALLOC(w->labels);
    CHECK_ALLOC(w->c);
    CHECK_ALLOC(w->inv_var);
    CHECK_ALLOC(w->acc);
}

static void workspace_free(Workspace *w)
{
    free(w->labels);
    free(w->c);
    free(w->inv_var);
    free(w->acc);
}

/* ----------------------------------------------------------------------------------------
 * FUSED PASS on the LOCAL rows: E-step + local sums of the M-step, in ONE parallel region
 *
 * Input:  local block ds, current parameters p (equal on all the ranks).
 * Output: w->acc = [Nk | sx | sx2] of the local rows (still to be summed over the ranks),
 *         w->labels, and the LOCAL log-likelihood (returned). No MPI call here.
 * ---------------------------------------------------------------------------------------- */
static double fused_pass(const Dataset *ds /* in */, const GMMParams *p /* in */,
                         Workspace *w /* in/out */)
{
    const int n = ds->N, D = ds->D, K = p->K;
    const int len = K + 2 * K * D;
    double *acc = w->acc;
    /* Local copies of the pointers: the compiler cannot prove that the stores into acc / buf do not
     * modify the fields of w (or the arrays they point to), so it would reload them at every
     * iteration and give up vectorizing. */
    const double *restrict X = ds->X;
    const double *restrict mu_all = p->mu;
    /* inv_var and cterm are written (cluster terms) and then read in the same function: with
     * restrict all the accesses must go through these pointers, so both are done through them */
    double *restrict inv_var = w->inv_var;
    double *restrict cterm = w->c;
    int *restrict labels = w->labels;

    /* The OpenMP reduction ADDS the private copies to the value already in acc: zero it first */
    memset(acc, 0, (size_t)len * sizeof(double));

    double ll = 0.0;
#pragma omp parallel reduction(+ : ll) reduction(+ : acc[:len])
    {
        /* My K values: a LOCAL array of this thread (on its own stack), so no cache line is
         * shared with another thread (no false sharing, see (3) at the top) */
        double buf[K];

        /* Cluster terms (K*D operations) shared among the threads. The implicit barrier at the
         * end of the "omp for" makes inv_var and c visible to everybody before the main loop. */
#pragma omp for schedule(static)
        for (int k = 0; k < K; k++)
        {
            double sum_log_var = 0.0;
            for (int d = 0; d < D; d++)
            {
                double v = p->var[(size_t)k * D + d];
                inv_var[(size_t)k * D + d] = 1.0 / v;
                sum_log_var += log(v);
            }
            cterm[k] = log(p->pi[k]) - 0.5 * (D * LOG_2PI + sum_log_var);
        }

        /* Main loop: for every row, E-step (responsibilities in buf) and immediately the update
         * of the sums. Every thread works on different rows and has private copies of acc and ll. */
#pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
        {
            const double *restrict x = X + (size_t)i * D;

            /* E-step of the row: log(pi_k N(x_i | mu_k, var_k)) for every k */
            for (int k = 0; k < K; k++)
            {
                const double *restrict mu = mu_all + (size_t)k * D;
                const double *restrict iv = inv_var + (size_t)k * D;
                double s = 0.0; /* squared Mahalanobis distance */
                for (int d = 0; d < D; d++)
                {
                    double diff = x[d] - mu[d];
                    s += diff * diff * iv[d];
                }
                buf[k] = cterm[k] - 0.5 * s;
            }

            /* Normalization (log-sum-exp); buf is overwritten with the responsibilities */
            double lse = log_sum_exp(buf, K);
            ll += lse;
            int best = 0;
            for (int k = 0; k < K; k++)
            {
                buf[k] = exp(buf[k] - lse);
                if (buf[k] > buf[best])
                    best = k;
            }
            labels[i] = best;

            /* M-step of the row: accumulate in the private copy of acc */
            for (int k = 0; k < K; k++)
            {
                double rik = buf[k];
                acc[k] += rik; /* Nk */
                double *restrict sx = acc + K + (size_t)k * D;
                double *restrict sx2 = sx + (size_t)K * D;
                for (int d = 0; d < D; d++)
                {
                    double rx = rik * x[d];
                    sx[d] += rx;
                    sx2[d] += rx * x[d];
                }
            }
        }
    } /* end of the parallel region: ll and acc are the sums of the private copies */
    return ll;
}

/* ----------------------------------------------------------------------------------------
 * M-step, part 2: new parameters from the GLOBAL sums in w->acc (same as the baseline)
 * Returns the number of clusters found empty.
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
 * the ranks). Every rank executes the same sequence of steps, because the convergence test uses
 * the GLOBAL log-likelihood.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    int iterations;   /* number of parameter updates performed */
    int converged;    /* 1 = tolerance reached, 0 = maximum number of iterations */
    int empty_events; /* clusters found empty (summed over the iterations) */
    double ll;        /* final GLOBAL log-likelihood, consistent with the final parameters */
    double t_estep;   /* seconds in the fused passes (this rank): E-step + local M-step sums */
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
    const int len = K + 2 * K * p->D;

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
static void warmup_em(const Dataset *ds, GMMParams *p, Workspace *w, int iters, long N_total)
{
    EMResult scratch; /* receives (and discards) the times */
    memset(&scratch, 0, sizeof scratch);
    const int K = p->K;
    const int len = K + 2 * K * p->D;
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

    /* Memory: parameters (replicated on every rank), working space, and the arrays used to
     * choose the K initial rows and to remember the labels of the best run. */
    GMMParams p, best;
    gmm_alloc(&p, K, D);
    gmm_alloc(&best, K, D);
    Workspace w;
    workspace_alloc(&w, n_local, K, D);
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
         * (the last one evaluates the log-likelihood). fused = 1: this version makes a single
         * pass over X per iteration and never stores the responsibilities. */
        st.flops = pem_flops(PEM_DIAGONAL, N_total, D, K, total_iterations + a.restarts, total_iterations);
        st.bytes = pem_bytes(1, N_total, D, K, total_iterations + a.restarts, total_iterations);
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
