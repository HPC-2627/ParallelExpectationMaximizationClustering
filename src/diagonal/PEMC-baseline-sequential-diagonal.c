/*
 * PEMC-baseline-sequential-diagonal.c
 *
 * Sequential EM for a mixture of Gaussians with DIAGONAL covariance.
 * It is the CORRECTNESS REFERENCE (all the other versions must reach the same final
 * log-likelihood starting from the same initialization) and the SERIAL TIME used to compute
 * speedup and efficiency.
 *
 * ========================================================================================
 * WHAT THE ALGORITHM DOES
 * ========================================================================================
 * Given N points (rows) with D variables (columns), it looks for K groups (clusters). Every group
 * k is a Gaussian with a mean mu_k, a variance for every variable var_k (diagonal covariance: the
 * variables are independent inside the group) and a weight pi_k (the share of points of the group).
 *
 * Model:  p(x) = sum_k pi_k * N(x | mu_k, diag(var_k))
 *
 * EM alternates two steps until the parameters no longer change:
 *
 *   E-step (Expectation): for every row i and cluster k it computes the RESPONSIBILITY, i.e. the
 *           probability that row i belongs to cluster k with the current parameters:
 *               r_ik = pi_k N(x_i | mu_k, var_k) / sum_j pi_j N(x_i | mu_j, var_j)
 *           It is computed in the logarithmic domain with log-sum-exp (no underflow).
 *           It also gives the log-likelihood  ll = sum_i log sum_k pi_k N(x_i | mu_k, var_k),
 *           which measures how well the model explains the data (the higher, the better).
 *
 *   M-step (Maximization): recomputes the parameters with the responsibilities as WEIGHTS (they are the
 *           usual means and variances, but every row counts in proportion to its probability of
 *           belonging to the cluster):
 *               N_k   = sum_i r_ik                    ("how many" rows the cluster has)
 *               mu_k  = sum_i r_ik x_i / N_k          (weighted mean)
 *               var_k = sum_i r_ik x_i^2 / N_k - mu_k^2   (weighted variance, E[x^2] - E[x]^2)
 *               pi_k  = N_k / N
 *           The M-step is expressed with SUMS over the rows (N_k, sum r x, sum r x^2): in the
 *           parallel versions every process computes them on its own rows and an MPI_Allreduce
 *           combines them.
 *
 * ========================================================================================
 * HOW THE DATA ARE LAID OUT IN MEMORY (everything row by row, "row-major")
 * ========================================================================================
 *   X[i*D + d]     variable d of row i              (N*D values)   rows are contiguous
 *   mu[k*D + d]    mean of variable d in cluster k             (K*D values)
 *   var[k*D + d]   variance of variable d in cluster k         (K*D values)
 *   r[i*K + k]     responsibility of row i for cluster k      (N*K values)
 * Indices are always computed with size_t: with large N and D, i*D exceeds the limit of an int.
 *
 * Cost of an iteration: the E-step and the M-step both do N*K*D operations, so
 * O(N*K*D). Memory: N*D (data) + N*K (responsibilities), because this version stores them.
 *
 * ========================================================================================
 * CHOICES OF THIS VERSION
 * ========================================================================================
 * It is the simple version, written for clarity: it stores the N x K matrix of responsibilities
 * and makes two separate passes over the data (E-step, M-step).
 *
 * Stopping: when |ll - ll_prev| <= tol * |ll| or after max_iter iterations.
 * The labels (if the dataset has them) are used only at the end, to compute the adjusted Rand
 * index: EM never sees them.
 *
 * Usage: see args_usage() (or run with -h).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "args.h"
#include "cluster-out.h"
#include "common.h"
#include "dataset.h"
#include "eval.h"
#include "gmm.h"
#include "logger.h"
#include "perfmodel.h"
#include "my-time-lib.h"

/* Name of the version: it ends up in the log and in the name of the default log file */
#define VERSION_NAME "baseline-sequential-diagonal"

/* ----------------------------------------------------------------------------------------
 * Working memory of EM
 *
 * All the arrays are allocated ONCE before starting and reused at every iteration:
 * allocating inside the loop would cost time and distort the measurement.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    double *r;       /* N*K: responsibilities, r[i*K + k]. It is the largest matrix after the data */
    double *c;       /* K: part of the log-density that depends only on the cluster */
    double *inv_var; /* K*D: 1 / var, computed once per iteration (a division per point would be slow) */
    double *logp;    /* K: log(pi_k N(x_i | ...)) of the current row */
    double *Nk;      /* K: sum_i r_ik */
    double *sx;      /* K*D: sum_i r_ik x_id */
    double *sx2;     /* K*D: sum_i r_ik x_id^2 */
} Workspace;

static void workspace_alloc(Workspace *w, int N, int K, int D)
{
    w->r = (double *)malloc((size_t)N * K * sizeof(double));
    w->c = (double *)malloc((size_t)K * sizeof(double));
    w->inv_var = (double *)malloc((size_t)K * D * sizeof(double));
    w->logp = (double *)malloc((size_t)K * sizeof(double));
    w->Nk = (double *)malloc((size_t)K * sizeof(double));
    w->sx = (double *)malloc((size_t)K * D * sizeof(double));
    w->sx2 = (double *)malloc((size_t)K * D * sizeof(double));
    /* if memory is missing (for example very large N) it terminates with a clear message */
    CHECK_ALLOC(w->r);
    CHECK_ALLOC(w->c);
    CHECK_ALLOC(w->inv_var);
    CHECK_ALLOC(w->logp);
    CHECK_ALLOC(w->Nk);
    CHECK_ALLOC(w->sx);
    CHECK_ALLOC(w->sx2);
}

static void workspace_free(Workspace *w)
{
    free(w->r);
    free(w->c);
    free(w->inv_var);
    free(w->logp);
    free(w->Nk);
    free(w->sx);
    free(w->sx2);
}

/* ----------------------------------------------------------------------------------------
 * E-step: responsibilities and log-likelihood
 *
 * Input:  data ds and current parameters p.
 * Output: fills w->r with the responsibilities and returns the log-likelihood ll.
 * ---------------------------------------------------------------------------------------- */
static double e_step(const Dataset *ds, const GMMParams *p, Workspace *w)
{
    const int N = ds->N, D = ds->D, K = p->K;

    /* Part 1: terms that depend ONLY on the cluster (not on the row).
     *
     * The log-density of a Gaussian with diagonal covariance, weighted with pi_k, is
     *   log pi_k - 0.5 * [ D log(2 pi) + sum_d log var_kd + sum_d (x_d - mu_kd)^2 / var_kd ]
     * The first piece (log pi_k, D log(2 pi), sum_d log var_kd) is the same for all the rows:
     * it is computed once per iteration and saved in c[k]. The same for 1 / var:
     * multiplying by the inverse is much faster than dividing N*K*D times. */
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

    /* Part 2: for every row, the K log-densities are computed, normalized and summed into ll */
    double ll = 0.0;
    for (int i = 0; i < N; i++)
    {
        const double *x = ds->X + (size_t)i * D; /* pointer to the beginning of row i */

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
            w->logp[k] = w->c[k] - 0.5 * s; /* log(pi_k N(x_i | mu_k, var_k)) */
        }

        /* Normalization with log-sum-exp: r_ik = exp(logp_k - lse), where
         * lse = log(sum_k exp(logp_k)) = log p(x_i). Working with logarithms prevents the
         * densities (tiny numbers with large D) from becoming 0 and giving 0/0 = NaN. */
        double lse = log_sum_exp(w->logp, K);
        ll += lse; /* contribution of this row to the log-likelihood */
        double *ri = w->r + (size_t)i * K;
        for (int k = 0; k < K; k++)
            ri[k] = exp(w->logp[k] - lse); /* the K responsibilities of the row sum to 1 */
    }
    return ll;
}

/* ----------------------------------------------------------------------------------------
 * M-step: new parameters from the responsibilities.
 * Returns the number of clusters found empty.
 * ---------------------------------------------------------------------------------------- */
static int m_step(const Dataset *ds, GMMParams *p, Workspace *w)
{
    const int N = ds->N, D = ds->D, K = p->K;

    /* The accumulators are zeroed (they still hold values of the previous iteration) */
    memset(w->Nk, 0, (size_t)K * sizeof(double));
    memset(w->sx, 0, (size_t)K * D * sizeof(double));
    memset(w->sx2, 0, (size_t)K * D * sizeof(double));

    /* Part 1: sufficient statistics, the three sums over the rows (the ones that the parallel
     * versions will combine with an MPI_Allreduce):
     *   Nk[k]      = sum_i r_ik
     *   sx[k][d]   = sum_i r_ik * x_id
     *   sx2[k][d]  = sum_i r_ik * x_id^2
     * A single pass over the rows computes all three. */
    for (int i = 0; i < N; i++)
    {
        const double *x = ds->X + (size_t)i * D;
        const double *ri = w->r + (size_t)i * K;
        for (int k = 0; k < K; k++)
        {
            double rik = ri[k];
            w->Nk[k] += rik;
            double *sx = w->sx + (size_t)k * D;
            double *sx2 = w->sx2 + (size_t)k * D;
            for (int d = 0; d < D; d++)
            {
                double rx = rik * x[d]; /* r_ik * x_id: reused for both sums */
                sx[d] += rx;
                sx2[d] += rx * x[d];
            }
        }
    }

    /* Part 2: from the sums to the new parameters (a small computation: K*D values, negligible
     * compared to the pass over the N rows) */
    int empty = 0;
    for (int k = 0; k < K; k++)
    {
        if (w->Nk[k] < NK_EMPTY)
        {
            /* Empty cluster: mean and variances stay as before (dividing by ~0 would give NaN) */
            empty++;
            p->pi[k] = PI_FLOOR;
            continue;
        }
        for (int d = 0; d < D; d++)
        {
            double mean = w->sx[(size_t)k * D + d] / w->Nk[k];
            /* variance = E[x^2] - E[x]^2. With standardized data (small values) the
             * subtraction is safe; with very large values it would lose significant digits */
            double var = w->sx2[(size_t)k * D + d] / w->Nk[k] - mean * mean;
            if (var < VAR_FLOOR) /* regularization (and protection against rounding errors) */
                var = VAR_FLOOR;
            p->mu[(size_t)k * D + d] = mean;
            p->var[(size_t)k * D + d] = var;
        }
        double pi = w->Nk[k] / N;
        p->pi[k] = pi < PI_FLOOR ? PI_FLOOR : pi; /* never 0, otherwise log(0) in the E-step */
    }
    return empty;
}

/* ----------------------------------------------------------------------------------------
 * A complete EM run starting from the parameters p (initialized by the caller)
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    int iterations;    /* number of M-steps performed */
    int converged;     /* 1 = tolerance reached, 0 = maximum number of iterations */
    int empty_events;  /* clusters found empty (summed over the iterations) */
    double ll;         /* final log-likelihood, consistent with the final parameters */
    double t_estep;    /* seconds in the E-step */
    double t_mstep;    /* seconds in the M-step */
    int n_hist;        /* values written in hist (see run_em) */
} EMResult;

/* hist (may be NULL): if given, it receives the log-likelihood after every E-step (room for
 * max_iter + 1 values), for the convergence plot C1; res.n_hist says how many were written. */
static EMResult run_em(const Dataset *ds, GMMParams *p, Workspace *w, const RunArgs *a,
                       double *hist)
{
    EMResult res;
    memset(&res, 0, sizeof res);
    double ll = -INFINITY, ll_prev = -INFINITY;

    /* One timer for each phase: the times accumulate over the iterations.
     * It measures real time (wall-clock); TIMER_ELAPSED returns microseconds. */
    TIMER_DEF(e);
    TIMER_DEF(m);

    /* Structure of every iteration:
     *   1. E-step with the current parameters -> responsibilities and ll
     *   2. if ll changed less than the tolerance compared to the previous iteration, STOP
     *   3. otherwise M-step: new parameters from the responsibilities
     * The check is made right after the E-step because ll is computed exactly there: this way no
     * extra E-step is needed to measure convergence. */
    for (int it = 0; it < a->max_iter; it++)
    {
        TIMER_START(e);
        ll = e_step(ds, p, w);
        TIMER_STOP(e);
        res.t_estep += TIMER_ELAPSED(e) / 1e6; /* microseconds -> seconds */
        if (hist != NULL)
            hist[res.n_hist++] = ll;

        if (a->verbose) /* printing is outside the timers (and only with -v) */
            printf("  iteration %3d  log-likelihood %.10g  change %+.3e\n", it, ll,
                   isfinite(ll_prev) ? ll - ll_prev : 0.0);

        if (gmm_converged(ll, ll_prev, a->tol))
        {
            res.converged = 1;
            break; /* the parameters are the ones that gave ll: nothing else to do */
        }

        TIMER_START(m);
        res.empty_events += m_step(ds, p, w);
        TIMER_STOP(m);
        res.t_mstep += TIMER_ELAPSED(m) / 1e6;

        res.iterations++;
        ll_prev = ll;
    }

    /* If the loop ended because of max_iter, the last operation was an M-step: ll and the
     * responsibilities refer to the previous parameters. One more E-step makes them
     * consistent. */
    if (!res.converged)
    {
        TIMER_START(e);
        ll = e_step(ds, p, w);
        TIMER_STOP(e);
        res.t_estep += TIMER_ELAPSED(e) / 1e6;
        if (hist != NULL)
            hist[res.n_hist++] = ll;
    }
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
 * It is called from main() in the warm-up pass (restart index -1), right after the
 * initialization; the parameters it modifies are then re-initialized with the SAME seed of
 * restart 0, so the final result is identical with and without warm-up.
 * ---------------------------------------------------------------------------------------- */
static void warmup_em(const Dataset *ds, GMMParams *p, Workspace *w, int iters)
{
    for (int it = 0; it < iters; it++)
    {
        e_step(ds, p, w);
        m_step(ds, p, w); /* the number of empty clusters returned is not needed here */
    }
}

/* ----------------------------------------------------------------------------------------
 * main
 *
 * Phases, in order:
 *   0. reading of the arguments
 *   1. reading of the dataset
 *   2. EM (one or more initializations)
 *   3. evaluation with the true labels (optional, outside the timing)
 *   4. log and summary on screen
 * ---------------------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    /* In short: read arguments -> load data -> for each restart: pick K rows as initial means and
     * run EM -> keep the best run -> compute ARI -> write the log and print the summary. */
    /* 0. Command-line arguments (-h prints the help and terminates successfully) */
    RunArgs a;
    args_defaults(&a, VERSION_NAME);
    int rc = args_parse(argc, argv, &a);
    if (rc != 0)
        return rc > 0 ? EXIT_SUCCESS : EXIT_FAILURE;

    /* Timers of the whole program and of the reading and initialization phases */
    TIMER_DEF(total);
    TIMER_DEF(read);
    TIMER_DEF(init);
    TIMER_DEF(tw); /* time of the warm-up pass */
    TIMER_START(total);

    /* 1. Reading of the dataset (the first n_rows rows of the file, or all of them).
     *    The reading time is measured separately: in the plots it serves to distinguish the
     *    cost of the data from the cost of EM. */
    TIMER_START(read);
    Dataset ds;
    if (dataset_load(a.dataset, 0, a.n_rows, &ds) != 0)
        return EXIT_FAILURE;
    TIMER_STOP(read);
    double t_read = TIMER_ELAPSED(read) / 1e6;

    const int N = ds.N, D = ds.D, K = a.K;
    if (K > N)
        DIE("the number of clusters (%d) is larger than the number of rows (%d)", K, N);

    /* Memory: current parameters (p), best parameters (best) and working space of EM.
     * idx and rows are used to choose the K initial rows; best_labels to remember the cluster
     * of every row in the best run. */
    GMMParams p, best;
    gmm_alloc(&p, K, D);
    gmm_alloc(&best, K, D);
    Workspace w;
    workspace_alloc(&w, N, K, D);
    uint64_t *idx = (uint64_t *)malloc((size_t)K * sizeof(uint64_t));
    double *rows = (double *)malloc((size_t)K * D * sizeof(double));
    int *best_labels = (int *)malloc((size_t)N * sizeof(int));
    CHECK_ALLOC(idx);
    CHECK_ALLOC(rows);
    CHECK_ALLOC(best_labels);
    /* Log-likelihood curve (only if --save): the current run and the best run so far */
    double *hist_cur = NULL, *hist_best = NULL;
    if (a.save_dir != NULL)
    {
        hist_cur = (double *)malloc((size_t)(a.max_iter + 1) * sizeof(double));
        hist_best = (double *)malloc((size_t)(a.max_iter + 1) * sizeof(double));
        CHECK_ALLOC(hist_cur);
        CHECK_ALLOC(hist_best);
    }

    /* 2. EM, with one or more initializations (restarts): the run with the best final
     *    log-likelihood is kept. With restarts = 1 (default) there is a single run.
     *    Why restarts: EM finds a LOCAL maximum of the likelihood, which depends on where it
     *    starts; starting from different points a better solution may be found.
     *    In the performance measurements a single restart is used, to get the time of one EM. */
    double t_init = 0.0, t_estep = 0.0, t_mstep = 0.0;
    double t_warmup = 0.0; /* seconds of the warm-up pass, removed from t_total */
    int have_best = 0, total_iterations = 0;
    EMResult best_res;
    memset(&best_res, 0, sizeof best_res);

    /* Restart index -1 is the WARM-UP pass (only if --warmup > 0): it initializes like restart 0,
     * runs a.warmup throwaway EM iterations (warmup_em) and goes back to the top of the loop.
     * The real restarts 0, 1, ... then start again from their own initialization, so the results
     * are the same as without warm-up. The warm-up time is measured apart (t_warmup) and
     * subtracted from t_total, and the init time of the warm-up pass is not added to t_init. */
    for (int rs = (a.warmup > 0 ? -1 : 0); rs < a.restarts; rs++)
    {
        const int warm = (rs < 0);
        if (warm)
            TIMER_START(tw);
        /* Initialization: K distinct rows chosen with the seeded generator become the
         * means (same procedure in all the versions: see gmm.h). The seed of every restart is
         * seed, seed+1, ...: same seed = same start = reproducible result. */
        TIMER_START(init);
        gmm_init_indices((uint64_t)N, K, a.seed + (uint64_t)(warm ? 0 : rs), idx);
        for (int k = 0; k < K; k++)
            memcpy(rows + (size_t)k * D, ds.X + (size_t)idx[k] * D, (size_t)D * sizeof(double));
        gmm_init_from_rows(&p, rows);
        TIMER_STOP(init);
        if (!warm)
            t_init += TIMER_ELAPSED(init) / 1e6;

        if (warm)
        {
            /* WARM-UP: throwaway iterations on the real data and workspace (see warmup_em) */
            warmup_em(&ds, &p, &w, a.warmup);
            TIMER_STOP(tw);
            t_warmup = TIMER_ELAPSED(tw) / 1e6;
            continue; /* the next iteration is restart 0: re-initialization, then the real run */
        }

        if (a.verbose)
            printf("run %d (seed %llu)\n", rs, (unsigned long long)(a.seed + (uint64_t)rs));
        EMResult res = run_em(&ds, &p, &w, &a, hist_cur);
        t_estep += res.t_estep;
        t_mstep += res.t_mstep;
        total_iterations += res.iterations;

        /* The best one is kept. The assignments are computed here because w.r holds the
         * responsibilities of the LAST run: after the next restart they would be lost. */
        if (!have_best || res.ll > best_res.ll)
        {
            have_best = 1;
            best_res = res;
            if (hist_cur != NULL)
                memcpy(hist_best, hist_cur, (size_t)res.n_hist * sizeof(double));
            gmm_copy(&best, &p);
            gmm_assign(w.r, N, K, best_labels); /* w.r refers to this run */
        }
    }
    TIMER_STOP(total);
    double t_total = TIMER_ELAPSED(total) / 1e6 - t_warmup; /* warm-up excluded */

    /* 3. Evaluation against the true labels (if the dataset has them): not part of the timing.
     *    The ARI is 1 if the clusters coincide with the true classes, about 0 if random. */
    double ari = NAN, pur = NAN;
    int32_t *truth = NULL;
    if (dataset_load_labels(a.dataset, 0, N, &truth) == 0)
    {
        ari = adjusted_rand_index(truth, best_labels, N, K);
        pur = purity(truth, best_labels, N, K);
    }

    /* Clustering result for the plots (labels, means, log-likelihood curve): only with --save */
    if (a.save_dir != NULL && cluster_save(a.save_dir, a.dataset, "diagonal", &best, best_labels, N,
                                            hist_best, best_res.n_hist) != 0)
        fprintf(stderr, "warning: the clustering result was not saved\n");

    /* 4. Log and summary.
     *    RunStats collects everything that goes into the log: the same format in all the versions. */
    RunStats st;
    memset(&st, 0, sizeof st);
    st.version = a.name;
    st.dataset = a.dataset;
    st.ranks = 1;   /* sequential version: one process, one thread */
    st.threads = 1;
    st.N = N;
    st.D = D;
    st.K = K;
    st.restarts = a.restarts;
    st.warmup_iters = a.warmup;
    st.t_warmup = t_warmup;
    st.iterations = best_res.iterations;
    st.converged = best_res.converged;
    st.loglik = best_res.ll;
    st.ari = ari;
    st.t_total = t_total;
    st.t_read = t_read;
    st.t_init = t_init;
    st.t_estep = t_estep;
    st.t_mstep = t_mstep;
    /* Analytic work for the roofline (P10): E-step runs iterations + 1 times per restart */
    st.flops = pem_flops(PEM_DIAGONAL, N, D, K, total_iterations + a.restarts, total_iterations);
    st.bytes = pem_bytes(0, N, D, K, total_iterations + a.restarts, total_iterations);
    st.t_comm = 0.0; /* no communication in the sequential version */

    if (a.log_path != NULL && log_run(a.log_path, &st) != 0)
        fprintf(stderr, "warning: the log line was not written\n");

    /* Summary on screen (after the measurements are over: printing during them would slow the program) */
    printf("=== %s ===\n", a.name);
    printf("dataset         : %s\n", a.dataset);
    printf("rows x columns  : %d x %d, K = %d\n", N, D, K);
    printf("seed / restarts : %llu / %d\n", (unsigned long long)a.seed, a.restarts);
    printf("warm-up         : %d iterations, %.4f s (not in the times)\n", a.warmup, t_warmup);
    printf("iterations      : %d (%s)\n", best_res.iterations, best_res.converged ? "converged" : "maximum number of iterations reached");
    printf("log-likelihood  : %.10g\n", best_res.ll);
    
    if (!isnan(ari) && ari > -1.5)
        printf("vs true labels  : ARI %.4f, purity %.4f\n", ari, pur);

    printf("time (s)        : total %.4f | read %.4f | init %.4f | E-step %.4f | M-step %.4f\n", t_total, t_read, t_init, t_estep, t_mstep);
    if (total_iterations > 0)
        printf("per iteration   : %.3f ms (E-step + M-step)\n",1e3 * (t_estep + t_mstep) / total_iterations);
    if (best_res.empty_events > 0)
        printf("warning         : %d empty cluster(s) found during the iterations\n", best_res.empty_events);
    if (a.log_path != NULL)
        printf("log             : %s\n", a.log_path);


    /* Release of the memory */
    free(truth);
    free(idx);
    free(rows);
    free(best_labels);
    free(hist_cur);
    free(hist_best);
    workspace_free(&w);
    gmm_free(&p);
    gmm_free(&best);
    dataset_free(&ds);
    return EXIT_SUCCESS;
}
