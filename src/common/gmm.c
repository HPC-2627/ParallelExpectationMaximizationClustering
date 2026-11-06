/*
 * gmm.c
 *
 * Management of the parameters of the Gaussian mixture (allocation, copy, initialization) and
 * two shared numerical functions (log-sum-exp, convergence test).
 * The meaning of each function and of its parameters is described in gmm.h.
 */

#include "gmm.h"
#include "common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
        uint64_t r = ((uint64_t)rand() << 31) ^ (uint64_t)rand();
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
