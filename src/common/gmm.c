/*
 * gmm.c
 *
 * Management of the parameters of the Gaussian mixture (allocation, copy, initialization),
 * shared numerical functions (log-sum-exp, convergence test), the assignment of the rows to the
 * clusters and, for the versions with full covariance, Cholesky factorization.
 * The meaning of each function and of its parameters is described in gmm.h. Same sections as
 * the header: shared by all the versions, diagonal covariance, full covariance.
 */

#include "gmm.h"
#include "common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================================
 * 1. SHARED BY ALL THE VERSIONS (diagonal and full)
 * ====================================================================================== */

void gmm_alloc(GMMParams *p, int K, int D)
{
    p->K = K;
    p->D = D;
    /* calloc = malloc + zeroing. The cast to size_t BEFORE the multiplication K*D avoids int
     * overflow with large values. */
    p->pi = (double *)calloc((size_t)K, sizeof(double));
    p->mu = (double *)calloc((size_t)K * D, sizeof(double));
    p->var = (double *)calloc((size_t)K * D, sizeof(double));
    CHECK_ALLOC(p->pi);
    CHECK_ALLOC(p->mu);
    CHECK_ALLOC(p->var);
}

void gmm_free(GMMParams *p)
{
    free(p->pi);
    free(p->mu);
    free(p->var);
    p->pi = p->mu = p->var = NULL; /* avoids using already released memory by mistake */
}

void gmm_copy(GMMParams *dst, const GMMParams *src)
{
    memcpy(dst->pi, src->pi, (size_t)src->K * sizeof(double));
    memcpy(dst->mu, src->mu, (size_t)src->K * src->D * sizeof(double));
    memcpy(dst->var, src->var, (size_t)src->K * src->D * sizeof(double));
}

void gmm_init_indices(uint64_t N, int K, uint64_t seed, uint64_t *idx)
{
    /* rand() depends only on the seed (srand): every version (and every MPI rank) running on
     * the same machine/C library obtains the same indices, without communicating.
     * rand() returns only 31 bits (RAND_MAX >= 2^31-1 on glibc), so two calls are combined to
     * be able to index datasets with more than 2^31 rows. Rejection sampling: a candidate is
     * accepted only if not already drawn (cheap because K is tiny compared to N). */
    srand((unsigned int)seed);
    int found = 0;
    while (found < K)
    {
        uint64_t hi = (uint64_t)rand();                            
        uint64_t lo = (uint64_t)rand();
        uint64_t r = (hi << 31) ^ lo;
        uint64_t candidate = r % N;
        int repeated = 0;
        for (int j = 0; j < found; j++)
        {
            if (idx[j] == candidate)
            {
                repeated = 1;
                break;
            }
        }
        if (!repeated)
            idx[found++] = candidate;
    }
}

void gmm_init_from_rows(GMMParams *p, const double *rows)
{
    /* means = the K chosen rows (copy of K*D values in one go, because rows has the same
     * layout as mu); variance 1 (standardized data); equal weights 1/K */
    memcpy(p->mu, rows, (size_t)p->K * p->D * sizeof(double));
    for (int k = 0; k < p->K; k++)
    {
        p->pi[k] = 1.0 / p->K;
        for (int d = 0; d < p->D; d++)
            p->var[(size_t)k * p->D + d] = 1.0;
    }
}

double log_sum_exp(const double *v, int K)
{
    /* Step 1: find the maximum of the vector */
    double max = v[0];
    for (int k = 1; k < K; k++)
        if (v[k] > max)
            max = v[k];

    /* Edge case: if the maximum is infinite (all the terms are -inf, or there is a +inf)
     * v[k] - max would give inf - inf = NaN. The correct result is the maximum itself. */
    if (isinf(max))
        return max;

    /* Step 2: the maximum is subtracted before exponentiating. The largest term becomes
     * exp(0) = 1 and the others are <= 1, so no overflow; the possible underflow of the
     * others (an exponential that equals 0) is harmless because the maximum alone contributes 1.
     * Identity used: log(sum exp(v)) = max + log(sum exp(v - max)). */
    double sum = 0.0;
    for (int k = 0; k < K; k++)
        sum += exp(v[k] - max);
    return max + log(sum);
}

int gmm_converged(double ll, double ll_prev, double tol)
{
    if (!isfinite(ll_prev)) /* first iteration: ll_prev is -inf, there is no previous value */
        return 0;
    /* absolute change <= tolerance * absolute value of ll (relative comparison) */
    return fabs(ll - ll_prev) <= tol * fabs(ll);
}

void gmm_assign(const double *r, int N, int K, int *labels)
{
    for (int i = 0; i < N; i++)
    {
        const double *ri = r + (size_t)i * K;
        int best = 0;
        for (int k = 1; k < K; k++)
            if (ri[k] > ri[best])
                best = k;
        labels[i] = best;
    }
}

/* ======================================================================================
 * 2. DIAGONAL COVARIANCE (only src/diagonal/)
 * ====================================================================================== */

/* Nothing else: see the note in gmm.h. */

/* ======================================================================================
 * 3. FULL COVARIANCE (only src/full/)
 * ====================================================================================== */

void gmm_full_init_identity(double *cov, int K, int D)
{
    /* All zeros, then 1 on the diagonal of every matrix (the cast to size_t BEFORE the
     * multiplication avoids int overflow with large K and D). */
    memset(cov, 0, (size_t)K * D * D * sizeof(double));
    for (int k = 0; k < K; k++)
        for (int d = 0; d < D; d++)
            cov[(size_t)k * D * D + (size_t)d * D + d] = 1.0;
}

int gmm_cholesky(const double *A, double *L, int D, double jitter)
{
    /* Row by row, left to right, only the lower triangle (j <= i). Every element of L depends
     * on the elements already computed in the same row (columns m < j) and in the row j. */
    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double s = A[(size_t)i * D + j];
            if (i == j)
                s += jitter; /* the jitter goes only on the diagonal */
            for (int m = 0; m < j; m++)
                s -= L[(size_t)i * D + m] * L[(size_t)j * D + m];
            if (i == j)
            {
                /* Pivot: it must be positive, otherwise the matrix is not positive definite.
                 * Written as !(s > 0) so that a NaN is rejected too. */
                if (!(s > 0.0))
                    return -1;
                L[(size_t)i * D + i] = sqrt(s);
            }
            else
                L[(size_t)i * D + j] = s / L[(size_t)j * D + j];
        }
    }
    return 0;
}

int gmm_full_factorize(const double *cov, const double *pi, int K, int D, double *L, double *c)
{
    for (int k = 0; k < K; k++)
    {
        const double *S = cov + (size_t)k * D * D; /* Sigma_k */
        double *Lk = L + (size_t)k * D * D;        /* its factor */

        /* If the factorization fails (rounding errors made Sigma_k not positive definite) a
         * jitter is added to the diagonal: VAR_FLOOR first, then 10 times larger each time. The
         * limit stops the loop for a matrix that cannot be repaired (e.g. it contains NaN). */
        double jitter = 0.0;
        while (gmm_cholesky(S, Lk, D, jitter) != 0)
        {
            jitter = jitter == 0.0 ? VAR_FLOOR : jitter * 10.0;
            if (jitter > 1e6)
                return k + 1;
        }

        /* log det Sigma = 2 sum_d log L_dd, because det Sigma = det(L)^2 and the determinant
         * of a triangular matrix is the product of its diagonal. */
        double log_det = 0.0;
        for (int d = 0; d < D; d++)
            log_det += 2.0 * log(Lk[(size_t)d * D + d]);
        c[k] = log(pi[k]) - 0.5 * (D * LOG_2PI + log_det);
    }
    return 0;
}
