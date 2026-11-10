/*
 * PEMC-baseline-sequential-full.c
 *
 * Sequential EM for a mixture of Gaussians with FULL covariance. Same structure as
 * PEMC-baseline-sequential-diagonal.c (read that one for the general description of EM, the
 * memory layout and the stopping rule); only what changes because of the covariance is
 * described here.
 *
 * MODEL:  p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)   with Sigma_k a full D x D matrix
 * (the variables of a cluster can be correlated, so the clusters can be tilted ellipsoids).
 *
 * WHAT CHANGES COMPARED TO THE DIAGONAL VERSION
 *   Memory:   cov[k*D*D + a*D + b] = Sigma_k[a][b]   (K*D*D values, row-major, symmetric)
 *             L[k*D*D + ...]       = Cholesky factor of Sigma_k (lower triangular), per iteration
 *
 *   E-step:   the log-density needs the squared Mahalanobis distance
 *                 (x-mu)^T Sigma^-1 (x-mu)    and    log det Sigma.
 *             Sigma is factorized ONCE per cluster and iteration as Sigma = L L^T (Cholesky):
 *                 log det Sigma = 2 sum_d log L_dd
 *                 (x-mu)^T Sigma^-1 (x-mu) = |z|^2   with  L z = x-mu  (forward substitution)
 *             No matrix inverse is ever formed. Cost per iteration: N*K*D^2/2 for the
 *             substitutions + K*D^3/6 for the factorizations (negligible when N >> D).
 *
 *   M-step:   the sufficient statistics become  Nk, sx (sum r x)  and  sxx[a][b] = sum r x_a x_b
 *             (K*D*D values; only the lower triangle is accumulated, N*K*D^2/2 operations):
 *                 Sigma_k = sxx / N_k - mu_k mu_k^T
 *             These are the sums that the parallel versions will combine with MPI_Allreduce.
 *
 *   Regularization: VAR_FLOOR is added to the diagonal of every Sigma_k (as in scikit-learn's
 *             reg_covar), which keeps it positive definite. If the Cholesky factorization still
 *             fails (rounding errors in E[xx^T] - mu mu^T) a growing jitter is added to the
 *             diagonal until it succeeds.
 *
 * Cost of an iteration: O(N*K*D^2) (diagonal: O(N*K*D)). Memory: N*D + N*K + K*D^2.
 *
 * Usage: see args_usage() (or run with -h).
 */

#include <math.h>
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
#include "my-time-lib.h"

/* Name of the version: it ends up in the log and in the name of the default log file */
#define VERSION_NAME "baseline-sequential-full"

/* ----------------------------------------------------------------------------------------
 * Working memory of EM
 *
 * All the arrays are allocated ONCE before starting and reused at every iteration:
 * allocating inside the loop would cost time and distort the measurement.
 * ---------------------------------------------------------------------------------------- */
typedef struct
{
    double *r;    /* N*K: responsibilities, r[i*K + k] */
    double *cov;  /* K*D*D: covariance matrices Sigma_k (full, symmetric, row-major) */
    double *L;    /* K*D*D: Cholesky factors of Sigma_k (lower triangle used), rebuilt every E-step */
    double *c;    /* K: part of the log-density that depends only on the cluster */
    double *z;    /* D: scratch vector of the forward substitution */
    double *logp; /* K: log(pi_k N(x_i | ...)) of the current row */
    double *Nk;   /* K: sum_i r_ik */
    double *sx;   /* K*D: sum_i r_ik x_id */
    double *sxx;  /* K*D*D: sum_i r_ik x_ia x_ib (lower triangle b <= a only) */
} Workspace;

static void workspace_alloc(Workspace *w, int N, int K, int D)
{
    w->r = (double *)malloc((size_t)N * K * sizeof(double));
    w->cov = (double *)malloc((size_t)K * D * D * sizeof(double));
    w->L = (double *)malloc((size_t)K * D * D * sizeof(double));
    w->c = (double *)malloc((size_t)K * sizeof(double));
    w->z = (double *)malloc((size_t)D * sizeof(double));
    w->logp = (double *)malloc((size_t)K * sizeof(double));
    w->Nk = (double *)malloc((size_t)K * sizeof(double));
    w->sx = (double *)malloc((size_t)K * D * sizeof(double));
    w->sxx = (double *)malloc((size_t)K * D * D * sizeof(double));
    CHECK_ALLOC(w->r);
    CHECK_ALLOC(w->cov);
    CHECK_ALLOC(w->L);
    CHECK_ALLOC(w->c);
    CHECK_ALLOC(w->z);
    CHECK_ALLOC(w->logp);
    CHECK_ALLOC(w->Nk);
    CHECK_ALLOC(w->sx);
    CHECK_ALLOC(w->sxx);
}

static void workspace_free(Workspace *w)
{
    free(w->r);
    free(w->cov);
    free(w->L);
    free(w->c);
    free(w->z);
    free(w->logp);
    free(w->Nk);
    free(w->sx);
    free(w->sxx);
}

/* ----------------------------------------------------------------------------------------
 * E-step: responsibilities and log-likelihood
 * ---------------------------------------------------------------------------------------- */
static double e_step(const Dataset *ds, const GMMParams *p, Workspace *w)
{
    const int N = ds->N, D = ds->D, K = p->K;

    /* Part 1: per-cluster terms: Cholesky factors of the Sigma_k and c[k] (see gmm.h, full covariance section) */
    int bad = gmm_full_factorize(w->cov, p->pi, K, D, w->L, w->c);
    if (bad)
        DIE("covariance of cluster %d is not positive definite", bad - 1);

    /* Part 2: for every row the K log-densities, log-sum-exp, responsibilities */
    double ll = 0.0;
    double *z = w->z;
    for (int i = 0; i < N; i++)
    {
        const double *x = ds->X + (size_t)i * D;

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
            w->logp[k] = w->c[k] - 0.5 * s;
        }

        double lse = log_sum_exp(w->logp, K);
        ll += lse;
        double *ri = w->r + (size_t)i * K;
        for (int k = 0; k < K; k++)
            ri[k] = exp(w->logp[k] - lse);
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

    memset(w->Nk, 0, (size_t)K * sizeof(double));
    memset(w->sx, 0, (size_t)K * D * sizeof(double));
    memset(w->sxx, 0, (size_t)K * D * D * sizeof(double));

    /* Part 1: sufficient statistics Nk, sx, sxx in a single pass over the rows.
     * sxx is symmetric: only the lower triangle (b <= a) is accumulated. */
    for (int i = 0; i < N; i++)
    {
        const double *x = ds->X + (size_t)i * D;
        const double *ri = w->r + (size_t)i * K;
        for (int k = 0; k < K; k++)
        {
            double rik = ri[k];
            w->Nk[k] += rik;
            double *sx = w->sx + (size_t)k * D;
            double *sxx = w->sxx + (size_t)k * D * D;
            for (int a = 0; a < D; a++)
            {
                double rx = rik * x[a];
                sx[a] += rx;
                double *row = sxx + (size_t)a * D;
                for (int b = 0; b <= a; b++)
                    row[b] += rx * x[b];
            }
        }
    }

    /* Part 2: from the sums to the new parameters */
    int empty = 0;
    for (int k = 0; k < K; k++)
    {
        if (w->Nk[k] < NK_EMPTY)
        {
            /* Empty cluster: mean and covariance stay as before */
            empty++;
            p->pi[k] = PI_FLOOR;
            continue;
        }
        const double inv = 1.0 / w->Nk[k];
        double *mu = p->mu + (size_t)k * D;
        for (int a = 0; a < D; a++)
            mu[a] = w->sx[(size_t)k * D + a] * inv;

        double *S = w->cov + (size_t)k * D * D;
        const double *sxx = w->sxx + (size_t)k * D * D;
        for (int a = 0; a < D; a++)
            for (int b = 0; b <= a; b++)
            {
                double v = sxx[(size_t)a * D + b] * inv - mu[a] * mu[b];
                if (a == b)
                    v += VAR_FLOOR; /* regularization: keeps Sigma positive definite */
                S[(size_t)a * D + b] = v;
                S[(size_t)b * D + a] = v; /* symmetric copy */
            }
        double pi = w->Nk[k] / N;
        p->pi[k] = pi < PI_FLOOR ? PI_FLOOR : pi;
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
} EMResult;

/* run_em is identical to the diagonal version: the loop does not care about the model, it
 * only calls e_step() and m_step(). Order: E-step -> test -> M-step, repeated. */
static EMResult run_em(const Dataset *ds, GMMParams *p, Workspace *w, const RunArgs *a)
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
    /* main is the same as in the diagonal version, with two differences: the workspace now holds
     * the K covariance matrices (w.cov), and after choosing the initial means the covariances
     * are set to the identity (cov_init_identity). Everything else (arguments, reading, restarts,
     * ARI, log, summary) is unchanged. */
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
        gmm_full_init_identity(w.cov, K, D);
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
        EMResult res = run_em(&ds, &p, &w, &a);
        t_estep += res.t_estep;
        t_mstep += res.t_mstep;
        total_iterations += res.iterations;

        /* The best one is kept. The assignments are computed here because w.r holds the
         * responsibilities of the LAST run: after the next restart they would be lost. */
        if (!have_best || res.ll > best_res.ll)
        {
            have_best = 1;
            best_res = res;
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
    st.t_comm = 0.0; /* no communication in the sequential version */

    if (a.log_path != NULL && log_run(a.log_path, &st) != 0)
        fprintf(stderr, "warning: the log line was not written\n");

    /* Summary on screen (after the measurements are over: printing during them would slow the program) */
    printf("=== %s ===\n", a.name);
    printf("dataset         : %s\n", a.dataset);
    printf("covariance      : full\n");
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
    workspace_free(&w);
    gmm_free(&p);
    gmm_free(&best);
    dataset_free(&ds);
    return EXIT_SUCCESS;
}
