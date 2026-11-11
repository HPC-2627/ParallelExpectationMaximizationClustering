/*
 * perfmodel.c
 *
 * FLOP and memory-traffic model of the EM. The conventions are in perfmodel.h; here every
 * formula is derived from the loops of the E-step and M-step of the versions.
 */

#include "perfmodel.h"

/* FLOP of ONE E-step. */
static double flops_estep(int model, long N, int D, int K)
{
    const double n = (double)N, d = (double)D, k = (double)K;
    /* Per row and per cluster, the squared Mahalanobis distance s and the log-density:
     *   diagonal: per variable  diff = x - mu (1), diff*diff*inv_var accumulated (2)  => 3*D
     *   full:     forward substitution L z = x - mu:
     *             row a has a products and a subtractions (2*a), plus x - mu (1), the division
     *             by L[a][a] (1) and z*z accumulated (2); summed over a: D*(D-1) + 4*D
     *             = D^2 + 3*D
     * then logp = c - 0.5*s (2 FLOP). */
    const double dist = (model == PEM_FULL) ? d * d + 3.0 * d : 3.0 * d;
    /* Per row, normalization: log-sum-exp (K subtractions of the max, K exp, K additions, 1 log,
     * 1 addition = 3K + 2) and r = exp(logp - lse) (2 per cluster) => about 5K + 2. */
    double f = n * (k * (dist + 2.0) + 5.0 * k + 2.0);
    /* Cholesky of the K matrices (once, see perfmodel.h): K * D^3 / 3 */
    if (model == PEM_FULL)
        f += k * d * d * d / 3.0;
    return f;
}

/* FLOP of ONE M-step. */
static double flops_mstep(int model, long N, int D, int K)
{
    const double n = (double)N, d = (double)D, k = (double)K;
    /* Per row and per cluster: Nk += r (1); per variable: rx = r*x (1), sx += rx (1) and the
     * second moments sxx += rx * x (2 FLOP each, diagonal: one per variable, full: lower
     * triangle only, a+1 entries in row a).
     *   diagonal: 1 + D * (1 + 1 + 2)         = 4*D + 1
     *   full:     1 + D * (1 + 1) + 2 * D(D+1)/2 = D^2 + 3*D + 1 */
    const double per = (model == PEM_FULL) ? d * d + 3.0 * d + 1.0 : 4.0 * d + 1.0;
    double f = n * k * per;
    /* Parameter update from the global sums (independent of N): small, but counted.
     * diagonal: mean (D) + variance (3*D) per cluster; full: D*(D+1)/2 entries, 3 FLOP each */
    f += (model == PEM_FULL) ? k * (d + 1.5 * d * (d + 1.0)) : k * 4.0 * d;
    return f;
}

double pem_flops(int model, long N, int D, int K, long n_estep, long n_mstep)
{
    return (double)n_estep * flops_estep(model, N, D, K) +
           (double)n_mstep * flops_mstep(model, N, D, K);
}

double pem_bytes(int fused, long N, int D, int K, long n_estep, long n_mstep)
{
    const double n = (double)N, d = (double)D, k = (double)K;
    const double x_pass = 8.0 * n * d; /* one pass over X (doubles) */
    const double r_pass = 8.0 * n * k; /* one pass over the responsibilities */
    if (fused)
        return (double)n_estep * x_pass; /* a single pass over X, r never stored; the M-step
                                          * sums are made inside the same pass */
    /* baseline: E-step reads X and writes r, M-step reads X and r */
    return (double)n_estep * (x_pass + r_pass) + (double)n_mstep * (x_pass + r_pass);
}
