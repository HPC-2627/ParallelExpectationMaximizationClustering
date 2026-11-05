/*
 * eval.c
 *
 * Adjusted Rand Index (ARI) and purity. Both are computed from the CONTINGENCY TABLE between
 * true classes and clusters found (see contingency()).
 * Meaning of the measures and input conventions: see eval.h.
 */

#include "eval.h"
#include "common.h"

#include <stdlib.h>

/* Maximum number of true classes accepted. Beyond this value the "label" is not a class but a
 * continuous value (for example a regression target): the contingency table would be huge and
 * the ARI would make no sense, so the computation is given up. */
#define MAX_CLASSES 4096

/* Builds the contingency table:
 *     n[c*K + k] = number of rows with true class c assigned to cluster k
 * (rows = true classes, columns = clusters found; row-major order like all the arrays).
 * Rows with a negative true label (outliers) are not counted.
 *   truth, pred, N, K   as in eval.h
 *   C   (output) number of true classes = maximum label + 1
 *   M   (output) number of rows with a valid label (>= 0), i.e. the sum of the table
 * Returns the table (allocated with calloc: the caller must free it), or NULL if it cannot be
 * built (no valid label, or too many classes). */
static long *contingency(const int32_t *truth, const int *pred, int N, int K, int *C, long *M)
{
    /* First pass: the maximum label gives the number of classes (labels are 0,1,2,...) */
    int max_label = -1;
    for (int i = 0; i < N; i++)
        if (truth[i] > max_label)
            max_label = truth[i];
    if (max_label < 0 || max_label >= MAX_CLASSES)
        return NULL;
    *C = max_label + 1;

    /* calloc zeroes the memory: all the cells start from 0 */
    long *n = (long *)calloc((size_t)(*C) * K, sizeof(long));
    CHECK_ALLOC(n);

    /* Second pass: every row increments the cell (true class, cluster found) */
    *M = 0;
    for (int i = 0; i < N; i++)
    {
        if (truth[i] < 0)
            continue; /* outlier: ignored */
        n[(size_t)truth[i] * K + pred[i]]++;
        (*M)++;
    }
    return n;
}

/* Number of distinct pairs that can be formed with x elements: x (x - 1) / 2.
 * Returns a double (not an integer) because with millions of rows x*(x-1)/2 exceeds the limit
 * of an int and, summed over many cells, also that of a long. */
static double choose2(double x)
{
    return x * (x - 1.0) / 2.0;
}

double adjusted_rand_index(const int32_t *truth, const int *pred, int N, int K)
{
    int C;
    long M;
    long *n = contingency(truth, pred, N, K, &C, &M);
    if (n == NULL || M < 2) /* at least 2 rows are needed to form a pair */
    {
        free(n); /* free(NULL) is harmless */
        return -2.0;
    }

    /* The "together" pairs (same group) are summed in three ways:
     *   sum_cells = pairs together in both the true and the found partition
     *               (sum of choose2 over every cell of the table)
     *   sum_rows  = pairs together in the TRUE partition
     *               (sum of choose2 of the row totals = rows of every true class)
     *   sum_cols  = pairs together in the FOUND partition
     *               (sum of choose2 of the column totals = rows of every cluster) */
    double sum_cells = 0.0, sum_rows = 0.0, sum_cols = 0.0;
    for (int c = 0; c < C; c++)
    {
        double row = 0.0; /* total of row c: rows of true class c */
        for (int k = 0; k < K; k++)
        {
            double v = (double)n[(size_t)c * K + k];
            sum_cells += choose2(v);
            row += v;
        }
        sum_rows += choose2(row);
    }
    for (int k = 0; k < K; k++)
    {
        double col = 0.0; /* total of column k: rows assigned to cluster k */
        for (int c = 0; c < C; c++)
            col += (double)n[(size_t)c * K + k];
        sum_cols += choose2(col);
    }
    free(n);

    /* ARI = (index - expected index) / (maximum index - expected index)
     *   index           = sum_cells
     *   expected index  = value obtained with two independent partitions
     *                     = sum_rows * sum_cols / choose2(M)
     *   maximum index   = mean of the two totals = 0.5 * (sum_rows + sum_cols)
     * So: identical partitions -> 1; independent partitions -> 0. */
    double expected = sum_rows * sum_cols / choose2((double)M);
    double maximum = 0.5 * (sum_rows + sum_cols);
    if (maximum == expected) /* both partitions are trivial (a single group): 0/0 */
        return 1.0;
    return (sum_cells - expected) / (maximum - expected);
}

double purity(const int32_t *truth, const int *pred, int N, int K)
{
    int C;
    long M;
    long *n = contingency(truth, pred, N, K, &C, &M);
    if (n == NULL || M < 1)
    {
        free(n);
        return -2.0;
    }
    /* For every cluster found (column of the table) the most frequent true class inside it is
     * taken: those rows are "correct". Purity is the share of correct rows. */
    long correct = 0;
    for (int k = 0; k < K; k++)
    {
        long best = 0; /* maximum of column k */
        for (int c = 0; c < C; c++)
            if (n[(size_t)c * K + k] > best)
                best = n[(size_t)c * K + k];
        correct += best;
    }
    free(n);
    return (double)correct / (double)M;
}
