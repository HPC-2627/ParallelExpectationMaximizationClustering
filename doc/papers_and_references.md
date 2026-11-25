# Papers and references

The source of each technique in our code, how close we follow it and whether it was checked on the paper. The `#` is the id cited in the code comments (e.g. `doc/papers_and_references.md #4.2`) and the prefix of the local PDF in `doc/papers/` (not versioned).

- **Strength** of the link: **direct** = same algorithm as the paper; **partial** = same idea, adapted; **own** = no paper, course principles or our measurements; **none** = related work cited for contrast, not implemented.
- **Status:** ✅ checked on the paper; ⚠️ weak source or related work only.
- `opt1` and `opt2` both start from `baseline-MPI-OpenMP` of their model and are compared with it.

---

## 1. References of our implementations

| # | Our code | Technique | Reference | Strength | Report key | Status |
|---|---|---|---|---|---|---|
| 1 | `baseline-sequential` ([diagonal](../src/diagonal/PEMC-baseline-sequential-diagonal.c), [full](../src/full/PEMC-baseline-sequential-full.c)) | EM for GMM: E/M formulas, monotonic log-likelihood, singularities | Bishop, *Pattern Recognition and Machine Learning*, 2006, ch. 9 · [page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) | direct | `bishop` | ✅ |
| 1b | all versions: `log_sum_exp` ([gmm.c](../src/common/gmm.c)), `VAR_FLOOR` ([common.h](../include/common.h)) | log-sum-exp, floor on the variances | Sanderson & Curtin 2017 (as `#2`), sec. 2.2; scikit-learn `reg_covar` · [arXiv:1707.09094](https://arxiv.org/abs/1707.09094), [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) | direct | `sanderson`, `sklearn` | ✅ |
| 2 | `baseline-MPI-OpenMP` ([diagonal](../src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c), [full](../src/full/PEMC-baseline-MPI-OpenMP-full.c)) | contiguous rows per thread, private accumulators, reduction | Sanderson & Curtin, ICSPCS 2017 · [arXiv:1707.09094](https://arxiv.org/abs/1707.09094), [DOI](https://doi.org/10.1109/ICSPCS.2017.8270510) | direct | `sanderson` | ✅ |
| 3.1 | — | distributed EM with frequent updates | Yin, Zhang, Gao, *JPDC* 111, 2018 · [DOI](https://doi.org/10.1016/j.jpdc.2017.07.005) | none | `yin` | ⚠️ related work |
| 3.2 | `baseline-MPI-OpenMP` | rows split across cores/nodes, partial sums | Lee, McLachlan, Leemaqz, arXiv 2020 · [arXiv:2005.06848](https://arxiv.org/abs/2005.06848) | partial | `lee2020` | ✅ |
| - | `baseline-MPI-OpenMP` | two `MPI_Allreduce` per iteration (`ll`, then the sums) | BYU CS584 student project, 1998 · [page](https://faculty.cs.byu.edu/~snell/Classes/CS584/projectsF98/serr/PEMpaper.html) | partial | `byu` | ⚠️ weak |
| 4.1 | `opt1` ([diagonal](../src/diagonal/PEMC-opt1-MPI-OpenMP-diagonal.c), [full](../src/full/PEMC-opt1-MPI-OpenMP-full.c)), point (1) | E-step and M-step fused in one pass, no $N \times K$ matrix $r$ | Lee, Leemaqz, McLachlan, arXiv 2016 · [arXiv:1606.02054](https://arxiv.org/abs/1606.02054) | partial | `lee2016` | ✅ |
| - | `opt1`, point (2) | one parallel region per iteration; full: the K Cholesky factorizations split among the threads | — | own | — | ✅ |
| - | `opt1`, point (3) | no false sharing: `buf[K]` on the stack / 64-byte aligned slices | — | own | — | ✅ |
| 4.2 | `opt1` full, point (4), blocks | blocks of $B = 64$ rows, Cholesky, sums in registers (level 2 → level 3) | Kwedlo, FedCSIS 2023 · [DOI](https://doi.org/10.15439/2023F9859), [PDF](https://annals-csis.org/Volume_35/drp/pdf/9859.pdf) | partial / own | `kwedlo` | ✅ |
| 4.4 | `opt1` full, point (4), SIMD | `#pragma omp simd` across rows (8 per tile, B per block) | Machlica, Vaněk, Zajíc, PDCAT 2011 · [DOI](https://doi.org/10.1109/PDCAT.2011.40), [PDF](https://kky-sw.zcu.cz/en/publications/1/LukasMachlica_2011_FastEstimationof.pdf) | partial | `machlica` | ✅ |
| 4.3a | `opt2` diagonal ([code](../src/diagonal/PEMC-opt2-MPI-OpenMP-diagonal.c)) | one `MPI_Allreduce`: `[Nk \| sx \| sx2 \| ll]` in one buffer | BYU project (above) + `#T` | partial | `byu` | ⚠️ weak (use `#T`) |
| 4.3b | `opt2` full ([code](../src/full/PEMC-opt2-MPI-OpenMP-full.c)) | one `MPI_Allreduce` + only the lower triangle of `sxx` | BYU project (above) + `#T` | partial + own | `byu` | ⚠️ weak (triangle: own) |
| T | `opt2` (and the communication of `baseline-MPI-OpenMP`) | cost of `MPI_Allreduce` vs $p$ and message size | Thakur, Rabenseifner, Gropp, *IJHPCA* 19(1), 2005 · [DOI](https://doi.org/10.1177/1094342005051521), [PDF](https://web.cels.anl.gov/~thakur/papers/mpi-coll.pdf) | direct (cost model) | `thakur` | ✅ |
| 4.5 | — | asynchronous distributed EM | Srivastava, DePalma, Liu, arXiv 2018 · [arXiv:1806.07533](https://arxiv.org/abs/1806.07533) | none | `srivastava` | ⚠️ related work |

### What we take and what differs

- **#1 Bishop.** E/M formulas (9.13)-(9.22), monotonic log-likelihood (sec. 9.2.2, 9.4), collapsing components (sec. 9.2.1). Differences: Bishop *resets* a collapsing component, we use a variance floor (`#1b`); we use the uncentered $\Sigma_k = \frac{1}{N_k}\sum_i r_{ik} x_i x_i^T - \mu_k\mu_k^T$ (`#2`, eq. 14), which needs only sums over the rows, so one pass and one `MPI_Allreduce`; log-sum-exp and Cholesky for the density are not in ch. 9.
- **#1b Sanderson, sec. 2.2.** Log domain (eq. 15-18) and a floor on the covariance diagonal. Differences: we subtract the maximum once ($m + \log\sum_k e^{v_k-m}$) instead of adding terms pairwise (eq. 18); in the full model we add the floor to the diagonal like scikit-learn `reg_covar` ($10^{-6}$), plus a growing jitter if Cholesky fails.
- **#2 Sanderson & Curtin.** Contiguous chunks of rows, per-thread $\sum r$, $\sum r x$, $\sum r x x^T$ (eq. 9-11) combined by reduction (eq. 12-14). Differences: the paper is OpenMP on one node, diagonal only, k-means initialization; we add MPI ranks, the full model and random-row initialization. Paper: $1\text{M} \times 100$, $K = 100$, 16 cores, $\sim 272 \to 27$ s.
- **#3.1 Yin et al.** Several parameter updates per pass (fewer passes). We keep one update per iteration, so the iterations and log-likelihood are identical to `baseline-sequential` and times can be compared.
- **#3.2 Lee 2020.** Data split into index sets, partial sums (sec. 3.2, Alg. 2). Differences: M-step split by component with a master thread, skew-$t$ mixtures in R/MATLAB, multi-node gains of only 2-4.5%.
- **BYU project.** Two `Allreduce` per iteration with contiguous buffers, like `baseline-MPI-OpenMP`. Student report on mixed-effects models (not GMM): historical note only.
- **#4.1 Lee 2016.** E-step results need not be collected before the M-step (sec. 4.3). Differences: the paper uses one thread per component; we fuse E+M per row/block, so $X$ is read once per iteration and $r$ is not stored.
- **#4.2 Kwedlo.** Cholesky, Mahalanobis distances and covariance sums on blocks of rows with level-3 operations to reuse the cache; 1.75-2.75× over level 2 on 24 cores (MKL). Differences: he factorizes the precision matrix and uses `dtrmm`/`dsyrk`, we factorize $\Sigma_k$ and use forward substitution with sums in registers; his blocks fit the L3, ours ($B = 64$) the L1/L2; single vs double precision; no E+M fusion (that is `#4.1`); Eigen + BLAS vs plain C. He lists hybrid MPI + OpenMP as future work, which is our setting.
- **#4.4 Machlica.** SIMD on the vector components without changing the formulas (sec. IV, Table I: 2.44-3.58× with SSE). Differences: GPU, diagonal, single precision; we vectorize in double across independent rows.
- **#4.3a / #4.3b opt2.** One `MPI_Allreduce` instead of two (the `ll` double is appended to the sums); in the full model only the lower triangle of `sxx` is sent: $K + KD + KD(D+1)/2 + 1$ doubles. The cost argument is `#T`.
- **#T Thakur.** Short messages: $T = \lg p\,\alpha + n \lg p\,(\beta + \gamma)$; long: $T \simeq 2\lg p\,\alpha + 2n\beta + n\gamma$. Removing the `ll` collective saves $\simeq \lg p\,\alpha$ per iteration: a large share when the sums are small (diagonal, $K + 2KD$ doubles, a few hundred bytes to a few KB), small on `gene_rnaseq` (≈ 1.6 MB). The triangle halves $n$: for $K = 6$, $D = 50$, 120 KB → 61 KB, where bandwidth counts.
- **#4.5 Srivastava.** The M-step starts when only part of the workers has answered. We are synchronous (the wait for the slowest rank is in `t_comm`): a possible extension.

---

## 2. Other references of the report

`Report key` above is the `\cite{key}` of [Group6-Report.tex](latex/Group6-Report.tex). The report also cites, not linked to the parallel code:

- `chandra` (Chandra et al., *Parallel Programming in OpenMP*, 2001): *Data Dependence Analysis*;
- `hubert` (adjusted Rand index, formula of [eval.c](../src/common/eval.c)): *Dataset*;
- `schwarz`, `akaike` (BIC and AIC of [clustering-summary.py](../src/tables/clustering-summary.py)): *Validation*, *Choice of $K$*.

Not in the bibliography: Foster's methodology is only named in *Foster's Design Strategy*; the datasets (UCI, OpenML, BrainWeb) are only linked in the table of the real datasets, with their DOIs and citations in [datasets.md](datasets.md), section 3.
