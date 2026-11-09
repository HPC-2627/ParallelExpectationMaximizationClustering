# Papers and references
## 1. Theory
| Reference | File | Content | Read |
|---|---|---|---|
| C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer 2006, ch. 9 (sections 9.2 and 9.3) | [1-bishop-prml.pdf](papers/1-bishop-prml.pdf) (official free PDF from the Microsoft Research website, complete 758-page book) | Mixture of Gaussians and EM, with the derivation of the formulas | [ ] |

## 2. EM on a single node
| Reference | File | What it does | Read |
|---|---|---|---|
| C. Sanderson, R. Curtin, "An Open Source C++ Implementation of Multi-Threaded Gaussian Mixture Models, k-Means and Expectation Maximisation", arXiv:1707.09094 | [2-sanderson-curtin-multithreaded-gmm.pdf](papers/2-sanderson-curtin-multithreaded-gmm.pdf) | Multi-threaded (OpenMP) EM for GMMs: each thread computes its own accumulators and a reduction combines them. On a synthetic dataset of about 762 MB (1 million samples, 100 dimensions, 100 components), with 16 cores the training time drops from about 272 s to about 27 s; the speedup is below linear because of OpenMP and reduction overhead and memory contention | [ ] |

## 3. EM on multiple nodes

| # | Reference | File | What it does | Read |
|---|---|---|---|---|
| 3.1 | J. Yin, Y. Zhang, L. Gao, "Accelerating distributed Expectation-Maximization algorithms with frequent updates", *J. Parallel Distrib. Comput.* 111 (2018) 65-75 | [3.1-yin-freem-frequent-updates.pdf](papers/3.1-yin-freem-frequent-updates.pdf) (copy published on an author's website) | Distributed framework for EM algorithms (k-means, fuzzy c-means, parameter estimation of a GMM, LDA), evaluated on a cluster of local machines and on Amazon EC2; with frequent updates it converges much faster than traditional implementations, while keeping the convergence properties of EM | [ ] |
| 3.2 | S. X. Lee, G. J. McLachlan, K. L. Leemaqz, "Multi-Node EM Algorithm for Finite Mixture Models", arXiv:2005.06848, 2020 | [3.2-lee-multinode-em.pdf](papers/3.2-lee-multinode-em.pdf) | Parallel EM for single-threaded and multi-threaded processors, on one machine and on multiple nodes, compared across R and MATLAB; it uses a fairly general mixture model (skewed distributions), not only Gaussians | [ ] |


## 4. Possible optimizations
| # | Optimization | Reference | File | What it says (checked on the text) | Same result as the baseline? | Read |
|---|---|---|---|---|---|---|
| 4.1 | Fusion of E-step and M-step | S. X. Lee, K. L. Leemaqz, G. J. McLachlan, "A simple multithreaded implementation of the EM algorithm for mixture models", arXiv:1606.02054, 2016 | [4.1-lee-multithreaded-em-mixture.pdf](papers/4.1-lee-multithreaded-em-mixture.pdf) | The results of the E-step need not be collected before the M-step: the M-step of a component can be performed straight after the E-step of the same component on the same thread, and the collection across threads happens only at the end of an iteration | yes | [ ] |
| 4.2 | Matrix formulation (level 3 BLAS) and cache blocking | W. Kwedlo, "Expectation-Maximization Algorithms for Gaussian Mixture Models Using Linear Algebra Libraries on Parallel Shared-Memory Systems", FedCSIS 2023 (*Proc. 18th Conf. on Computer Science and Intelligence Systems*, vol. 35, pp. 1047-1052, DOI 10.15439/2023F9859) | [4.2-kwedlo-2023-em-linear-algebra.pdf](papers/4.2-kwedlo-2023-em-linear-algebra.pdf) | Four OpenMP variants, 25 large datasets, two 12-core Xeon processors. The variant using matrix-matrix operations and L3 cache blocking is the fastest: 1.75 to 2.75 times faster than the naive version using matrix-vector operations, with a parallel efficiency always above 83% | yes | [ ] |
| 4.3 | Less communication: contiguous arrays, a single `MPI_Allreduce` | BYU course project (CS584), "A parallel implementation of the Expectation Maximization Algorithm" (year inferred from the website address: fall 1998) | [4.3-byu-parallel-em.html](papers/4.3-byu-parallel-em.html) (web page) | Two communication points per iteration, with `Allreduce`; allocating the 5 arrays of temporary results contiguously removes 8 log₂ n network latencies from the `Allreduce` sections. Tested on 1, 2, 4 and 8 Pentium Pro processors at 233 MHz | yes | [ ] |
| 4.4 | SIMD vectorization (SSE instructions) | L. Machlica, J. Vaněk, Z. Zajíc, "Fast Estimation of Gaussian Mixture Model Parameters on GPU using CUDA" (University of West Bohemia; year 2011 from the file name, venue to be verified) | [4.4-machlica-gmm-gpu-sse.pdf](papers/4.4-machlica-gmm-gpu-sse.pdf) | Besides the GPU version, it proposes a CPU version with SSE instructions, "significantly faster" than the naive one. On 3.1 million vectors of dimension 40: the GPU is about 150 times faster than the CPU with SSE and more than 400 times faster than the naive CPU, so SSE is worth about 2.7 times (ratio derived from these two numbers). No approximation in the formulas | yes | [ ] |
| 4.5 | Update with a fraction of the data at each iteration | S. Srivastava, G. DePalma, C. Liu, "An Asynchronous Distributed Expectation Maximization Algorithm For Massive Data: The DEM Algorithm", arXiv:1806.07533, 2018 | [4.5-srivastava-dem-asynchronous.pdf](papers/4.5-srivastava-dem-asynchronous.pdf) | The workers perform the E-step in parallel and the managers perform the M-step after receiving the results from a fraction γ of the workers; the sequence still converges to a local mode. Empirically it is more than 2 times faster than the non-distributed version for γ around 0.5 | **no**, it changes the algorithm | [ ] |

Notes:
- **4.3 is not about Gaussian mixture clustering.** It is an EM for two-level mixed-effects models (variance components), in a course project from the late 1990s. The only idea that transfers is merging several arrays into a single buffer to reduce the latency of collective communications. As a source it is weak: use it together with the MPI documentation and the course slides.
- **4.4** is mainly a GPU work: the part of interest here is the CPU version with SSE. SSE is an old SIMD extension (128-bit registers): today AVX2 or AVX-512 would be used, but the principle is the same.
- **4.5** was evaluated on linear mixed-effects models, not on GMMs, and should be cited as related work rather than as a recipe to copy.
- The **OpenMP reductions with per-thread accumulators** (a useful idea for the baseline and for optimization 4.1) are in the paper of section 2.
