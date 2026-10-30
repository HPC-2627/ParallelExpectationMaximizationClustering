# Questions for the instructors

1. **Dataset and use case.** Are we free to choose any dataset we want, and therefore to pick our own use case? We would like to focus on medical data.

2. **Real vs. synthetic data.** We plan to use about 10 real datasets (as many as possible from the medical domain, the others from different domains if needed) and 3 synthetic datasets with different characteristics, in small, medium and large sizes, for example to run weak scaling experiments. Would this be acceptable?

3. **Sequential reference, baseline and optimizations.** We plan to first write a sequential C implementation, used as the correctness reference (same final log-likelihood) and as the serial time for speedup and efficiency. Then we would build a baseline that already uses both MPI and OpenMP, implement 3 optimizations of that baseline (each one compared against the baseline), and test everything on both the real and the synthetic datasets. Is this an appropriate approach?

4. **Cluster resources.** What is the maximum number of nodes, cores and MPI ranks we can use for the single-node and multi-node benchmarks, and is there a recommended queue or a job time limit?

5. **Plots and metrics.** Which plots and metrics would you expect to see in the report? Is there anything else you would like us to include?

   5.1 Performance:
   - Time per iteration vs. number of processes/threads (log-log), one line per version (baseline and the 3 optimizations).
   - Speedup vs. number of processes, with the ideal line and an Amdahl's-law fit.
   - Parallel efficiency vs. number of processes, one curve per dataset size (small / medium / large).
   - Strong scaling (fixed problem size) and weak scaling (problem size grows with the resources).
   - Stacked bars of computation vs. communication (E-step, M-step, `MPI_Allreduce`) per iteration.
   - Heatmap of time per iteration for different MPI ranks × OpenMP threads combinations at a fixed total number of cores.
   - Point-to-point communication benchmark (bandwidth vs. message size, log scale) with a latency/bandwidth model fit, used to predict the cost of the `MPI_Allreduce`.
   - Speedup of each of the 3 optimizations over the baseline (bar chart, same resources and dataset).
   - Error bars (standard deviation over repeated runs, after discarding warm-up iterations) on all timing plots.
   - Load imbalance: minimum, mean and maximum time per MPI rank.
   - Effect of the PBS placement strategy (`pack`, `scatter`, `pack:excl` and `scatter:excl`) on the multi-node runs.
   - GFLOPS and memory bandwidth compared with the peak of the machine (roofline), to explain the gain of the memory-oriented optimizations.

   5.2 Clustering results (to show that the algorithm works on the data):
   - 2D PCA projection of the data colored by cluster, with the Gaussian ellipses and the point transparency given by the membership probability.
   - Heatmap of the cluster centroids over the (normalized) clinical variables.
   - Cluster sizes (mixing weights).
   - Log-likelihood vs. iterations (convergence curve), which also shows that all the versions (sequential, baseline, optimizations) give the same result.
   - BIC/AIC vs. number of clusters K, to justify the choice of K.
   - Distribution of each clinical variable per cluster (violin or box plots), since the centroid heatmap only shows the means.
   - Histogram of the maximum membership probability, to show how confident the assignments are.
   - If the chosen dataset has a known clinical label (e.g. diagnosis): comparison between clusters and label (contingency table or adjusted Rand index).
