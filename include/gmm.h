#ifndef GMM_H
#define GMM_H

/*
 * gmm.h
 *
 * GMM = Gaussian Mixture Model.
 * Parameters of the model and functions shared by the versions with DIAGONAL covariance
 * (src/diagonal/) and with FULL covariance (src/full/). The file has three sections:
 *   1. SHARED BY ALL THE VERSIONS (diagonal and full): parameters, initialization, log-sum-exp,
 *      convergence test, assignment of the rows to the clusters.
 *   2. DIAGONAL COVARIANCE: what the diagonal versions use on top of section 1.
 *   3. FULL COVARIANCE: what the full versions use on top of section 1.
 *
 * The loops of the E-step and of the M-step are NOT here: they are written in every version,
 * because they are exactly what the optimizations modify (loop fusion, vectorization, ...).
 *
 * MODEL
 *   p(x) = sum_k pi_k * N(x | mu_k, Sigma_k),     k = 0..K-1
 * i.e. the density of a point x is a weighted sum of K Gaussians. Diagonal covariance: inside
 * every cluster the D variables are independent (Sigma_k = diag(var_k): every variable has its
 * own mean and variance, no correlations). Full covariance: Sigma_k is a complete D x D matrix.
 *
 * MEMORY LAYOUT (all the arrays are doubles, in row-major order):
 *   pi[k]            mixing weight of cluster k (share of the data of the cluster; sum = 1)
 *   mu[k*D + d]      mean of variable d in cluster k
 *   var[k*D + d]     variance of variable d in cluster k       (diagonal versions only)
 *   cov[k*D*D + a*D + b]   Sigma_k[a][b], full symmetric matrix  (full versions only; kept in
 *                    the workspace of every version, NOT in GMMParams)
 *   L[k*D*D + a*D + b]     Cholesky factor of Sigma_k (lower triangle b <= a only)
 */

/* ======================================================================================
 * 1. SHARED BY ALL THE VERSIONS (diagonal and full)
 * ====================================================================================== */

#include <stdint.h>

typedef struct
{
    int K;       /* number of clusters */
    int D;       /* number of variables */
    double *pi;  /* K values */
    double *mu;  /* K*D values */
    double *var; /* K*D values: used only by the diagonal versions (the full ones keep Sigma_k in
                    their workspace and ignore this field) */
} GMMParams;

/* Allocates (and zeroes) the arrays of p for K clusters and D variables. Terminates the
 * program if memory is missing. */
void gmm_alloc(GMMParams *p, int K, int D);

/* Releases the arrays of p and sets them to NULL. */
void gmm_free(GMMParams *p);

/* Copies all the parameters of src into dst. dst must already be allocated with the same K
 * and D. */
void gmm_copy(GMMParams *dst, const GMMParams *src);

/*
 * INITIALIZATION, identical in all the versions
 * (EM finds a local maximum: starting from different points it can reach different results,
 * so to compare the versions they must all start from the same point).
 *
 * Step 1: gmm_init_indices() chooses K DISTINCT rows of the dataset (indices in [0, N)) with
 *         rand() seeded with the given seed. Only N, K and the seed are needed, not the data:
 *         every MPI rank can therefore compute the same indices without communicating (then
 *         the rank that owns a chosen row sends it to the others).
 *           N     total number of rows of the dataset
 *           K     number of clusters = how many rows to choose
 *           seed  seed of the generator
 *           idx   (output) array of K distinct indices
 * Step 2: gmm_init_from_rows() builds the parameters from the K chosen rows (rows, K*D values,
 *         row k at rows[k*D ...]):
 *           mu_k = chosen row k,    var = 1,    pi = 1/K.
 *         Variance 1 is the right order of magnitude because the data are standardized
 *         (z-score: mean 0, variance 1); this way no pass over the data is needed (nor
 *         communication between the processes).
 */
void gmm_init_indices(uint64_t N, int K, uint64_t seed, uint64_t *idx);
void gmm_init_from_rows(GMMParams *p, const double *rows);

/* log(sum_k exp(v[k])) for a vector v of K values, computed without overflow/underflow.
 * The trick: the maximum m is subtracted from all the terms, log(sum exp(v)) = m + log(sum exp(v-m)).
 * This way the largest term becomes exp(0) = 1 and the sum is never 0 nor infinite.
 * Computing it directly with exp(v[k]) would give 0 for very negative v (log-densities with
 * large D are hundreds of units below zero) and therefore log(0) = -inf. */
double log_sum_exp(const double *v, int K);

/* Convergence test on the log-likelihood: |ll - ll_prev| <= tol * |ll|
 * (the change is RELATIVE, so the threshold does not depend on the scale of the data).
 * Returns 1 if converged, 0 otherwise. It always returns 0 if ll_prev is not a finite number,
 * i.e. at the first iteration, when there is no previous value yet. */
int gmm_converged(double ll, double ll_prev, double tol);

/* Most probable cluster of every row: labels[i] = argmax_k r[i*K + k], for N rows with
 * responsibilities r (N*K values, row-major). In the MPI versions N is the number of LOCAL rows.
 * Used to compare the result with the true labels (ARI). */
void gmm_assign(const double *r, int N, int K, int *labels);

/* ======================================================================================
 * 2. DIAGONAL COVARIANCE (only src/diagonal/)
 * ====================================================================================== */

/* Nothing else is needed: the initial variances (var = 1) are set by gmm_init_from_rows() and
 * the variances are stored in GMMParams.var. */

/* ======================================================================================
 * 3. FULL COVARIANCE (only src/full/)
 * ====================================================================================== */

/* Initial covariances: Sigma_k = identity for all the K clusters (the data are standardized;
 * it corresponds to var = 1 of the diagonal version). cov has K*D*D values. */
void gmm_full_init_identity(double *cov, int K, int D);

/*
 * Cholesky factorization A = L L^T of a symmetric D x D matrix. Only the lower triangle of A
 * is read; `jitter` is added to the diagonal. L (D*D values) is written in its lower triangle.
 * Returns 0 on success, -1 if the matrix is not positive definite (a pivot <= 0).
 */
int gmm_cholesky(const double *A, double *L, int D, double jitter);

/*
 * Per-cluster terms of the E-step, for all the K clusters:
 *   L[k]  = Cholesky factor of cov[k]. If cov[k] is not positive definite (rounding errors),
 *           a jitter VAR_FLOOR, then 10x larger each time, is added to its diagonal until the
 *           factorization succeeds.
 *   c[k]  = log pi_k - 0.5 (D log 2pi + log det Sigma_k),   with log det = 2 sum_d log L_dd.
 * The log-density of a row x is then  c[k] - 0.5 |z|^2  with  L z = x - mu_k  (forward
 * substitution): no matrix inverse is ever formed.
 *   cov  K*D*D (in)    pi  K (in)    L  K*D*D (out)    c  K (out)
 * Returns 0 on success, or k+1 if cluster k could not be factorized even with a jitter of 1e6
 * (the caller decides how to terminate: DIE in a sequential program, MPI_Abort in an MPI one).
 * It is deterministic: ranks with equal parameters take the same decisions.
 */
int gmm_full_factorize(const double *cov, const double *pi, int K, int D, double *L, double *c);


#endif
