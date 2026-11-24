#ifndef EVAL_H
#define EVAL_H

/*
 * eval.h
 *
 * Quality of the clustering with respect to the TRUE labels (when the dataset has them).
 *
 * Clustering is unsupervised: EM finds the groups without knowing the true classes. If the
 * true classes are known (synthetic datasets, or labeled real ones) one can measure how closely
 * the groups found reproduce them. These measures serve two purposes:
 *   - evaluating the quality of the result;
 *   - checking that ALL the versions (baseline-sequential, baseline-MPI-OpenMP, opt1, opt2) find the same groups:
 *     if the ARI changes between two versions, there is an error somewhere.
 * The labels are NEVER used by EM itself.
 *
 * Rows with a NEGATIVE true label (the outliers of the synthetic datasets) are ignored in all
 * the computations: they do not belong to any class, so evaluating them makes no sense.
 *
 * Input conventions common to the two functions:
 *   truth  N true labels (int32); classes are numbered 0, 1, 2, ...; negative = outlier
 *   pred   N clusters found by EM, numbered in [0, K)
 *   N      number of rows; K number of clusters
 * The numbering of the clusters is arbitrary (EM's "cluster 2" may correspond to the true
 * "class 0"): this is why the measures compare partitions, not numbers.
 */

#include <stdint.h>

/* Adjusted Rand Index (ARI).
 * Measures the agreement between two partitions by counting the PAIRS of rows that the two
 * partitions treat in the same way (together in both, or separated in both), corrected for
 * chance: a random assignment gives ARI ~ 0.
 *   1      = identical partitions (up to the numbering of the groups)
 *   ~ 0    = agreement equal to the random one (can be slightly negative)
 * Returns -2.0 if it cannot be computed (no valid label, fewer than 2 valid rows, or too many
 * classes, i.e. the "target" is continuous and not made of classes). */
double adjusted_rand_index(const int32_t *truth, const int *pred, int N, int K);

/* Purity: share of rows that belong to the MAJORITY true class of their predicted cluster.
 * For example, a cluster with 90 rows of class A and 10 of class B contributes 90 "correct"
 * rows. It is easier to read than the ARI, but has a flaw: it grows with the number of
 * clusters (with one cluster per row it would be 1). Returns -2.0 if it cannot be computed. */
double purity(const int32_t *truth, const int *pred, int N, int K);

#endif
