# Synthetic datasets
## 1. Why synthetic data

- Real medical datasets are small (hundreds to a few thousand rows): with so little data the communication weighs more than the computation and no scaling is visible. The weak scaling needs a problem size that grows with the number of processes, which is possible only with generated data.
- With generated data the truth is known (true subgroup of each patient and true parameters): all the versions (sequential, baseline, optimizations) can be checked against each other, and the quality of the clustering can be measured.
- Three profiles cover different situations: an easy case that satisfies the model, a realistic case (overlapping groups, correlated variables) and a hard case (rare group, outliers, many variables). $K$ and $D$ change too, because the cost of an iteration is roughly proportional to $N \cdot K \cdot D$.

## 2. The three profiles

| | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|
| Role | easy, satisfies the model | realistic | hard |
| Clusters $K$ | 4 | 5 | 6 |
| Variables $D$ | 10 | 20 | 50 |
| Cluster weights | 25%, 25%, 25%, 25% | 30%, 25%, 20%, 15%, 10% | 40%, 25%, 18%, 12%, 4%, 1% |
| Separation $\delta$ (distance between cluster means, in standard deviations) | 6.0 (very well separated) | 2.5 (overlapping) | 4.0 (intermediate) |
| Covariance inside a cluster | diagonal | correlated (factor model, strength 0.5, 3 factors) | diagonal |
| Standard deviation per variable | 1.0 | 1.0 (before correlation) | drawn in [0.6, 1.4] per variable and cluster |
| Outliers | 0% | 0% | 2% (label $-1$) |
| Skewed (log-normal) variables | no | yes (7 of the 20 clinical variables) | no |
| Physiological limits (clipping) | no | yes | yes (clinical variables only) |
| What it tests | correctness check, reference for scaling | slow convergence, choice of $K$, mismatch of the diagonal model | numerical stability: rare group, outliers, high $D$ |

Notes:

- In `synth_overlap` the variables are correlated inside each cluster (for example age with pressure, BMI with glucose), so the **diagonal covariance model is slightly wrong** for these data. This is intentional and has to be stated as a limit.
- In `synth_rare` the imbalance of the cluster sizes does **not** unbalance the work between the MPI ranks (the cost of a point is the same for all points): this profile tests numerical stability (empty or collapsing clusters, underflow, regularization), not load balancing.
- The EM cost per iteration is roughly $N \cdot K \cdot D$, so the three profiles differ in the amount of work per rank (tables in sections 4 and 5).

## 3. Variables

The first 20 variables are clinical, with typical values of an adult population (they are the base values used to turn the generated values into clinical units; the final files are standardized, see section 6). For $D > 20$ the remaining columns are generic biomarkers.

| # | Variable | Unit | Typical mean ± std | Physiological limits | Skewed |
|---|---|---|---|---|---|
| 1 | `age` | years | 55 ± 15 | 18 – 100 | no |
| 2 | `bmi` | kg/m² | 27 ± 5 | 14 – 60 | no |
| 3 | `sys_bp` | mmHg | 130 ± 20 | 80 – 230 | no |
| 4 | `dia_bp` | mmHg | 80 ± 11 | 40 – 130 | no |
| 5 | `glucose` | mg/dL | 105 ± 30 | 50 – 400 | yes |
| 6 | `hba1c` | % | 5.8 ± 1.0 | 4 – 14 | yes |
| 7 | `cholesterol` | mg/dL | 200 ± 40 | 90 – 400 | no |
| 8 | `hdl` | mg/dL | 52 ± 14 | 20 – 110 | no |
| 9 | `creatinine` | mg/dL | 1.0 ± 0.3 | 0.3 – 6 | yes |
| 10 | `heart_rate` | beats/min | 75 ± 12 | 40 – 150 | no |
| 11 | `ldl` | mg/dL | 120 ± 35 | 30 – 300 | no |
| 12 | `triglycerides` | mg/dL | 140 ± 70 | 30 – 800 | yes |
| 13 | `albumin` | g/dL | 4.2 ± 0.4 | 2 – 5.5 | no |
| 14 | `hemoglobin` | g/dL | 13.5 ± 1.8 | 6 – 19 | no |
| 15 | `wbc` | 10³/µL | 7.0 ± 2.0 | 2 – 25 | yes |
| 16 | `platelets` | 10³/µL | 250 ± 60 | 50 – 600 | no |
| 17 | `sodium` | mmol/L | 140 ± 3 | 120 – 160 | no |
| 18 | `potassium` | mmol/L | 4.2 ± 0.4 | 2.5 – 6.5 | no |
| 19 | `alt` | U/L | 28 ± 15 | 5 – 200 | yes |
| 20 | `crp` | mg/L | 3.0 ± 3.0 | 0.1 – 100 | yes |
| 21–50 | `biomarker_01` … `biomarker_30` | — | 0 ± 1 | none | no |

Variables used by each profile: `synth_clean` uses 1–10, `synth_overlap` uses 1–20, `synth_rare` uses 1–50. "Skewed" variables are log-normal only in `synth_overlap`; limits apply only where clipping is on (`synth_overlap`, `synth_rare`).

The typical values are generic adult values chosen by us, **not taken from a real cohort**.

## 4. Weak scaling: files, sizes and work per rank

Every profile has one file per number of processes $p$. With $N_0 = 250\,000$ rows per process, the file for $p$ processes has $N = N_0 \cdot p$ rows. Rank $r$ reads rows from $r \cdot N_0$ to $(r+1) \cdot N_0 - 1$, so every rank always has $N_0$ rows. The files of a profile are nested: the file for $2p$ contains the rows of the file for $p$ plus $N_0$ new rows (same random stream), so going from $p$ to $2p$ only the amount of data changes, not the sample.

| $p$ | Rows $N = N_0 \cdot p$ | `synth_clean` data file | `synth_overlap` data file | `synth_rare` data file | Labels file (each profile) |
|---|---|---|---|---|---|
| 1 | 250 000 | 20 MB | 40 MB | 100 MB | 1 MB |
| 2 | 500 000 | 40 MB | 80 MB | 200 MB | 2 MB |
| 4 | 1 000 000 | 80 MB | 160 MB | 400 MB | 4 MB |
| 8 | 2 000 000 | 160 MB | 320 MB | 800 MB | 8 MB |
| 16 | 4 000 000 | 320 MB | 640 MB | 1 600 MB | 16 MB |
| **Total** | | **620 MB** | **1 240 MB** | **3 100 MB** | **31 MB** |

Total on disk with the default parameters: about 5 GB (the sum of the files of a profile is about twice the largest one: 1+2+4+8+16 = 31 versus 16). Sizes are $8 + N \cdot D \cdot 8$ bytes for the data (MB = $10^6$ bytes) and $N \cdot 4$ bytes for the labels.

Work per rank (constant in the weak scaling, since each rank always has $N_0$ rows):

| | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|
| $N_0 \cdot K \cdot D$ (proportional to the cost of an iteration) | $1.0 \cdot 10^7$ | $2.5 \cdot 10^7$ | $7.5 \cdot 10^7$ |
| Data held by one rank | 20 MB | 40 MB | 100 MB |

The cost per iteration of `synth_rare` is therefore about 7.5 times that of `synth_clean` per rank, which gives different regimes (memory-bound for small $D$, closer to compute-bound for large $D$). This is an operation count and not a measured time.

Ideal behaviour: the time stays constant when $p$ grows; efficiency $E = T_1/T_p$.

## 5. Strong scaling: fixed file and work per rank

The problem size is fixed: the largest file of each profile (`_p16`, $N = 4\,000\,000$ rows) is run with $p \in \{1,2,4,8,16\}$ processes. Rank $r$ reads $N/p$ rows, from $r \cdot N/p$ to $(r+1) \cdot N/p - 1$, so the work per rank decreases when $p$ grows. No extra file is needed.

| $p$ | Rows per rank $N/p$ | `synth_clean` data per rank | `synth_overlap` data per rank | `synth_rare` data per rank |
|---|---|---|---|---|
| 1 | 4 000 000 | 320 MB | 640 MB | 1 600 MB |
| 2 | 2 000 000 | 160 MB | 320 MB | 800 MB |
| 4 | 1 000 000 | 80 MB | 160 MB | 400 MB |
| 8 | 500 000 | 40 MB | 80 MB | 200 MB |
| 16 | 250 000 | 20 MB | 40 MB | 100 MB |

Work per rank, $N \cdot K \cdot D / p$ (proportional to the cost of an iteration):

| $p$ | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|
| 1 | $1.6 \cdot 10^8$ | $4.0 \cdot 10^8$ | $1.2 \cdot 10^9$ |
| 2 | $8.0 \cdot 10^7$ | $2.0 \cdot 10^8$ | $6.0 \cdot 10^8$ |
| 4 | $4.0 \cdot 10^7$ | $1.0 \cdot 10^8$ | $3.0 \cdot 10^8$ |
| 8 | $2.0 \cdot 10^7$ | $5.0 \cdot 10^7$ | $1.5 \cdot 10^8$ |
| 16 | $1.0 \cdot 10^7$ | $2.5 \cdot 10^7$ | $7.5 \cdot 10^7$ |

Ideal behaviour: the time halves when $p$ doubles; speedup $S = T_1/T_p$, efficiency $E = S/p$. With $p = 16$ the work per rank equals that of the weak scaling ($N_0 \cdot K \cdot D$), so the two experiments meet at the same point. These are operation counts and not measured times.


## 6. How the data are generated

1. **True model.** For every profile the true parameters are drawn once from a random generator with a fixed seed: the means $\mu_k$ and the covariances $\Sigma_k$ of the $K$ clusters.
   - Means: every coordinate is Gaussian with standard deviation $\delta / \sqrt{2D}$, so the distance between two cluster means is about $\delta$ (in standard deviations inside the cluster) whatever $D$ is. This gives $\delta$ the same meaning for all the profiles.
   - Correlated covariance (`synth_overlap`): random correlation matrix with a factor model, $C = I + 0.5 \cdot B B^T$ with $B$ of size $D \times 3$, normalized to a correlation matrix. Few factors give a low-rank structure, typical of clinical variables that depend on a few common causes (age, weight, metabolic state). Every cluster has its own matrix.
2. **Patients.** For each patient the subgroup is drawn with the weights, then the point is drawn from the Gaussian of that subgroup.
3. **Outliers** (`synth_rare`): 2% of the points are replaced by Gaussian noise with standard deviation 4.0 (3–4 standard deviations from the center of the data) and labelled $-1$: clearly anomalous but not absurd.
4. **Clinical units.** The generated values are converted with the base values of section 3: skewed variables become log-normal (`value = mean · exp((std/mean) · z)`), the others are `mean + std · z`; with clipping on, values are limited to the physiological interval.
5. **Standardization.** Every column is standardized (z-score), because the EM works on normalized data. Mean and standard deviation are estimated on a pilot sample of 200 000 rows and are the same for all the blocks and all the files of a profile (they are saved in the `.json`).

Every block of rows uses its own random generator derived from (seed, profile, block number), so the same seed and parameters always give identical files (seed 42 by default).

## 7. How to read data in C/C++

```c
FILE *f = fopen("data/synthetic/synth_clean/synth_clean_p04.bin", "rb");
int N, D;
fread(&N, sizeof(int), 1, f);
fread(&D, sizeof(int), 1, f);
double *X = malloc((size_t)N * D * sizeof(double));
fread(X, sizeof(double), (size_t)N * D, f);     // X[i*D + d] = variable d of patient i
fclose(f);
```

Why this format:

- **Binary and not CSV:** it is read with one `fread`, without converting text into numbers, and it is much smaller (8 bytes per value instead of 15–20 characters). The reading time is part of the measurements.
- **`double`:** the EM sums many numbers and computes `exp` and `log`; double precision avoids accuracy losses that would make the parallel versions differ.
- **Row-major order:** the rows of a rank are a contiguous block, so every rank can read only its own rows (`fseek` or MPI-IO).
- **Little-endian:** the byte order of the x86 processors of the cluster nodes.
- **8-byte header:** it describes the file and keeps the doubles aligned to 8 bytes.
- **Labels in a separate file:** the EM cannot "see" the truth and the data file stays a clean matrix of numbers.

## 8. Limitations to state in the report

- The values are generic adult values chosen by us, not from a real cohort: the datasets reproduce the **structure** (subgroups, correlations, skewness, outliers) of clinical data, not real patients. There are no privacy issues.
- `synth_overlap` is generated with correlated variables, while the baseline uses a diagonal covariance: the model does not match the data on purpose.
- Nested files: the weak scaling datasets of different $p$ share the same first rows. This makes the comparison cleaner, but the measurements for different $p$ are not independent samples.
- The work per rank is an operation count ($N_0 \cdot K \cdot D$), not a measured time.
