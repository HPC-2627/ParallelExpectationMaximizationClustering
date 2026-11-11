#ifndef CLUSTER_OUT_H
#define CLUSTER_OUT_H

/*
 * cluster-out.h
 *
 * Saves the CLUSTERING RESULT of a run (not the timings: those go to the .log, see logger.h),
 * so that src/plots-generate/plots-cluster.py can draw the clustering plots (C1-C6) without
 * running EM again. Enabled by the option --save DIR (see args.h).
 *
 * For a run on dataset "heart_failure" with the model "diagonal" and K = 2, three files are
 * written in DIR (the base name is <dataset file name without extension>_<model>_k<K>):
 *
 *   heart_failure_diagonal_k2.labels.bin   N int32: cluster of every row (0 .. K-1), same row
 *                                          order as the dataset file
 *   heart_failure_diagonal_k2.params.csv   header "cluster,pi,mu_0,...,mu_{D-1}", one row per
 *                                          cluster: weight and mean (in the SAME standardized
 *                                          units as the .bin file). Covariances are not saved
 *                                          (the plots only need the means)
 *   heart_failure_diagonal_k2.loglik.csv   header "iteration,loglik": log-likelihood after
 *                                          every E-step of the BEST run (convergence curve)
 *
 * Only the sequential baseline saves them: the result is the same in every version (the final
 * log-likelihood is checked), so one version is enough.
 */

#include "gmm.h"

/* Writes the three files above. 'model' is "diagonal" or "full" (used in the file names).
 * hist has n_hist values (the log-likelihood after each E-step); it may be NULL (no loglik file).
 * Missing folders of 'dir' are created. Returns 0 if ok, -1 on error (a message is printed). */
int cluster_save(const char *dir, const char *dataset, const char *model, const GMMParams *p,
                 const int *labels, int N, const double *hist, int n_hist);

#endif
