#ifndef PERFMODEL_H
#define PERFMODEL_H

/*
 * perfmodel.h
 *
 * Analytic count of the floating-point operations (FLOP) and of the main-memory traffic (bytes)
 * of an EM run. They are written in the log (keys "flops" and "bytes") and used by
 * src/plots-generate/plots-performance.py for the roofline plot (P10):
 *
 *     arithmetic intensity = flops / bytes          (FLOP per byte)
 *     performance          = flops / (t_estep + t_mstep)   (FLOP per second)
 *
 * Why an analytic count and not hardware counters: hardware counters (perf, PAPI, likwid) are not
 * available on every cluster node and they count what the compiler generated (vectorized code,
 * exp/log implementations), which is not comparable between versions. Counting the operations
 * of the ALGORITHM gives the same "useful work" for every version of the same model, so a faster
 * version really shows a higher performance (FLOP/s) and not just a different instruction mix.
 *
 * Counting conventions (one operation = one FLOP):
 *   - add, subtract, multiply, divide = 1 FLOP; a fused multiply-add is counted as 2;
 *   - exp() and log() are counted as 1 FLOP each (so the count is a LOWER bound of the real
 *     work: the real cost of exp/log is ~10-20 FLOP);
 *   - the Cholesky factorization of the full model is counted ONCE per E-step, although every
 *     MPI rank repeats it (the repetition is a cost of the parallel design, not useful work);
 *   - communication is not counted (it is not arithmetic).
 *
 * Memory traffic convention (bytes moved between the main memory and the CPU):
 *   - the data X (N*D doubles) and the responsibilities r (N*K doubles) do not fit in the
 *     cache, so every pass over them reads or writes them from the main memory;
 *   - the parameters (means, variances/covariances, Cholesky factors: K*D or K*D^2 doubles)
 *     are assumed to stay in the cache: their traffic is ignored. For the full model with
 *     large K*D^2 this is optimistic.
 *   - baseline-MPI-OpenMP: E-step reads X and WRITES r; M-step reads X and r again  => 2 passes.
 *   - fused (an optimization that does E-step and M-step sums in one pass and never stores r):
 *     X is read once per iteration and r is never written.
 *
 * The E-step of one EM run is executed (iterations + 1) times per restart (the last one only
 * evaluates the log-likelihood), the M-step iterations times: the callers pass the two counts.
 */

#define PEM_DIAGONAL 0 /* model: diagonal covariance */
#define PEM_FULL 1     /* model: full covariance */

/* FLOP of the whole run: n_estep E-steps and n_mstep M-steps over N rows (the rows of ALL the
 * ranks), D variables, K clusters. */
double pem_flops(int model, long N, int D, int K, long n_estep, long n_mstep);

/* Bytes moved from/to the main memory by the whole run (see the convention above).
 * fused = 0 for baseline-MPI-OpenMP (two passes, responsibilities stored), 1 for a fused version. */
double pem_bytes(int fused, long N, int D, int K, long n_estep, long n_mstep);

#endif
