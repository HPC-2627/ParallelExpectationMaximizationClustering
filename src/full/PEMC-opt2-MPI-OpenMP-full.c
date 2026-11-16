/*
 * PEMC-opt2-MPI-OpenMP-full.c
 *
 * OPTIMIZATION 2 of the hybrid MPI + OpenMP EM, FULL covariance: COMMUNICATION.
 * NOT WRITTEN YET: this file is a placeholder and is not in the makefile (MPI_BINS) nor in the
 * versions of src/script/pemc-submit.sh (WRITTEN). Add it back there when the code is written.
 *
 * Plan (same idea as src/diagonal/PEMC-opt2-MPI-OpenMP-diagonal.c, read it first):
 *   - start from PEMC-opt1-MPI-OpenMP-full.c (all the optimizations of the computation: fused
 *     E+M pass, one parallel region per iteration, blocks of B rows) and change ONLY the
 *     communication, so that "opt2 vs opt1" measures the communication alone;
 *   - ONE MPI_Allreduce per iteration instead of two: the local log-likelihood travels in the last
 *     slot of the same buffer as the M-step sums, [Nk | sx | sxx | ll]; when the test says
 *     "converged", the sums of that iteration are simply not applied (as in the diagonal opt2);
 *   - sxx sent PACKED: only the lower triangle, K*D(D+1)/2 doubles instead of K*D*D (opt1 already
 *     accumulates only the lower triangle, but reduces the whole matrix, half of it zeros);
 *     unpacked after the Allreduce, before m_step_update.
 *
 * Why (logs of opt1 full with blocks, exclusive nodes): on the large datasets the communication is
 * only 1-2% of the time even on 4 nodes, but on the small ones (cardiotocography, heart_failure) it
 * reaches ~30% on 4 nodes (latency and waiting at every Allreduce): there one collective instead of
 * two, and half the bytes, can be measured.
 */
