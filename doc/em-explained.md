# EM clustering explained from scratch

Study guide, no machine learning prerequisites. The formulas render well in the VS Code Markdown preview.

## 1. The idea in short

You have a table of data (for example patients, each described by age, blood pressure, glucose...). **Nobody told you which group each patient belongs to**, but you suspect that groups exist.

The EM hypothesis: each group has a "bell shape" (a **Gaussian**) around a center, and the data are a **mixture** of these bells.

The problem is a chicken-and-egg one:
- if you knew the **groups** (center, shape, weight), you could compute for each patient **how likely it is that they belong to each one**;
- if you knew the **membership** of each patient, you could compute the **center and shape** of each group.

You know neither. EM starts from a random guess and alternates two steps until nothing changes anymore:

1. **E-step (Expectation):** with the current groups, compute for each point the probability of belonging to each group.
2. **M-step (Maximization):** with those probabilities, recompute center, shape and weight of each group.

It is a "soft" k-means: each point has a **probability** for each group ("70% A, 25% B, 5% C") instead of a hard label.

## 2. What data is needed

A **matrix** $X$ of real numbers, with $N$ rows and $D$ columns:

```
              age   BMI   pressure  glucose   ...   (D columns = variables)
patient 1     54    27.1    130       95
patient 2     61    31.4    145      130
patient 3     33    22.0    115       88
...
patient N     ...                                  (N rows = points)
```

- Each **row** is a point (a patient) and each **column** is a variable.
- Only **numeric** data is needed. A categorical variable (sex, smoker yes/no) must be turned into numbers (0/1 or one-hot).
- **No labels are needed**: it is unsupervised learning. If the dataset has a diagnosis, it is used only at the end for evaluation, not for training.
- **Normalization (z-score):** subtract the mean and divide by the standard deviation of each column. Without it, a variable with large numbers (cholesterol in mg/dL) weighs more than one with small numbers (BMI) only because of the scale.
- **Missing values:** they must be handled beforehand (remove the row or replace with the mean), because the algorithm does not accept them.
- **What you choose:** the number of groups $K$.

## 3. All the symbols

| Symbol | What it is | Size |
|---|---|---|
| $N$ | number of points (patients) | number |
| $D$ | number of variables (columns) | number |
| $K$ | number of groups (clusters) we look for | number |
| $i = 1 \dots N$ | point index | |
| $k = 1 \dots K$ | group index | |
| $d = 1 \dots D$ | variable index | |
| $x_i$ | point $i$, that is a row of $X$ | vector of $D$ numbers |
| $\mu_k$ | **mean** (center) of group $k$ | vector of $D$ numbers |
| $\Sigma_k$ | **covariance matrix** of group $k$ (shape) | $D \times D$ matrix |
| $\sigma^2_{kd}$ | variance of variable $d$ in group $k$ (diagonal version) | number |
| $\pi_k$ | **weight** of group $k$: share of points that belong to it | number between 0 and 1, sums to 1 |
| $\theta$ | all the parameters together: $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^{K}$ | |
| $z_i$ | "true" (hidden) group of point $i$: the **latent variable** | one of $1 \dots K$ |
| $r_{ik}$ | **responsibility**: probability that point $i$ belongs to group $k$ | number between 0 and 1, sums to 1 for each $i$ |
| $N_k$ | "effective number" of points of group $k$ | number |
| $\mathcal{N}(x \mid \mu, \Sigma)$ | Gaussian density | number $\ge 0$ |
| $\mathcal{L}$ | **log-likelihood**: how well the parameters explain the data | number |

## 4. Statistics concepts needed

### Mean
The central value. For variable $d$: $\bar{x}_d = \frac{1}{N}\sum_{i=1}^{N} x_{id}$.

### Variance and standard deviation
Measures **how much the values spread** around the mean:
$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2, \qquad \sigma = \sqrt{\sigma^2}$$
Small variance = values close to the mean (narrow bell). Large = widely spread (wide bell).

### Covariance (the question: what does it do?)
Variance looks at **one** variable. Covariance looks at **two** variables together and tells whether they **grow together**:
$$\mathrm{cov}(a, b) = \frac{1}{N}\sum_{i=1}^{N}(a_i - \bar{a})(b_i - \bar{b})$$

- **Positive:** when $a$ is above its mean, $b$ tends to be as well (age and pressure).
- **Negative:** when one goes up, the other goes down.
- **Close to zero:** no linear relationship.

The value depends on the units of measure, so the **correlation** is often used: the covariance divided by the two standard deviations (always between -1 and +1).

### Covariance matrix $\Sigma$ ($D \times D$)
It collects **all** variances and covariances. For two variables, age and BMI:

$$\Sigma = \begin{pmatrix} \mathrm{var}(\text{age}) & \mathrm{cov}(\text{age},\text{BMI}) \\ \mathrm{cov}(\text{BMI},\text{age}) & \mathrm{var}(\text{BMI}) \end{pmatrix}$$

- **Diagonal:** the variances of each variable.
- **Off-diagonal:** the covariances between pairs. It is **symmetric**.
- **What it does in the Gaussian:** it decides the **shape and orientation** of the bell. With zero covariance the bell is a circle (or an ellipse aligned with the axes). With positive covariance it is a **tilted** ellipse, because the two variables go together.

### Full and diagonal covariance
| | Full | Diagonal |
|---|---|---|
| Matrix | complete $D \times D$ | only the diagonal (vector of $D$ variances) |
| Group shape | tilted ellipse | ellipse aligned with the axes |
| Parameters per group | about $D^2/2$ | $D$ |
| Cost per iteration | $O(NKD^2)$ + inversions $O(KD^3)$ | $O(NKD)$ |
| Risk | unstable with high $D$ and few data | assumes independent variables inside the group |

**For the project it is better to start from the diagonal**: simpler formulas and no matrix inverses.

### The Gaussian
**In one dimension**, the bell with mean $\mu$ and variance $\sigma^2$:
$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- $(x-\mu)^2$: squared distance from the center. The farther the point, the more negative the exponent and the lower the value.
- $\sigma^2$ in the denominator: if the bell is wide, the distance "weighs less".
- The factor in front is only there so that the total area is 1.

**In $D$ dimensions**, with covariance matrix $\Sigma$:
$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{D/2}\sqrt{\det\Sigma}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$
- $(x-\mu)^T \Sigma^{-1} (x-\mu)$ is the squared **Mahalanobis distance**: a distance from the center that takes the shape of the group into account (being far along the short axis counts more than along the long axis).
- $\det\Sigma$ is the "volume" of the group and $\Sigma^{-1}$ the inverse: these are the expensive parts to compute.

**Diagonal case:** the Gaussian becomes the product of $D$ 1D Gaussians, one per variable, and the logarithm is a simple sum:
$$\log\mathcal{N}(x \mid \mu_k, \sigma_k^2) = -\frac{1}{2}\sum_{d=1}^{D}\left[\log(2\pi\sigma_{kd}^2) + \frac{(x_d-\mu_{kd})^2}{\sigma_{kd}^2}\right]$$

## 5. The model: Gaussian mixture (GMM)

The density of any point is a **weighted sum** of $K$ Gaussians:
$$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \sum_{k}\pi_k = 1$$

The "story" with which the data are imagined to have been generated:
1. A group $z$ is chosen at random: group $k$ with probability $\pi_k$ (the weight).
2. The point is drawn from a Gaussian with mean $\mu_k$ and covariance $\Sigma_k$.

We only see the point $x_i$, never the group $z_i$ (this is why it is a **latent variable**, that is hidden). EM is used to estimate $\theta$ without knowing $z_i$.

## 6. What we want to maximize: the likelihood

We want the parameters $\theta$ for which the observed data are **most probable**. The logarithm is used, because it turns products into sums and avoids tiny numbers:
$$\mathcal{L}(\theta) = \sum_{i=1}^{N}\log\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x_i \mid \mu_k, \Sigma_k)$$

- The higher it is (less negative), the better the model explains the data.
- There is no closed formula for the best parameters, because there is a sum inside the logarithm. Hence EM.
- **Guarantee:** at each iteration $\mathcal{L}$ does not decrease, so it converges to a **local** maximum (not necessarily the best one).

## 7. The algorithm, step by step

### Initialization
Starting values are needed for $\pi_k$, $\mu_k$, $\Sigma_k$ (see section 9).

### E-step: computing the responsibilities
$$r_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K}\pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

It reads like this: "the probability that point $i$ belongs to group $k$ is proportional to **how much the group weighs** ($\pi_k$) times **how well its bell explains the point** ($\mathcal{N}$)". The denominator normalizes, so for each point the $K$ probabilities sum to 1. This is Bayes' theorem.

### M-step: updating the parameters
Each formula is the usual mean or variance, with the points **weighted** by $r_{ik}$ instead of counted as whole.

| Parameter | Formula | In words |
|---|---|---|
| Effective points | $N_k = \sum_{i=1}^{N} r_{ik}$ | sum of the probabilities: "how many" points the group has (can be fractional) |
| Weight | $\pi_k = \dfrac{N_k}{N}$ | share of points of the group |
| Mean | $\mu_k = \dfrac{1}{N_k}\sum_{i} r_{ik}\,x_i$ | mean of the points, weighted by the probability of belonging to the group |
| Full covariance | $\Sigma_k = \dfrac{1}{N_k}\sum_{i} r_{ik}\,(x_i-\mu_k)(x_i-\mu_k)^T$ | weighted covariance |
| Variance (diagonal) | $\sigma_{kd}^2 = \dfrac{1}{N_k}\sum_{i} r_{ik}\,(x_{id}-\mu_{kd})^2$ | weighted variance, variable by variable |

A point with $r_{ik}=0.9$ counts "almost fully" in group $k$. One with $r_{ik}=0.01$ almost does not count.

### Convergence check
$\mathcal{L}$ is computed. If $|\mathcal{L}_{\text{new}}-\mathcal{L}_{\text{old}}|$ is below a tolerance (for example $10^{-6}$ relative), or the maximum number of iterations is reached, it stops. Otherwise it goes back to the E-step.

## 8. Numerical example by hand (1 dimension)

Data: $x = [1,\ 2,\ 5,\ 8,\ 9]$, so $N=5$, $D=1$ (the covariance becomes a simple variance), $K=2$.
Start: $\mu_1=0$, $\mu_2=10$, $\sigma_1^2=\sigma_2^2=4$, $\pi_1=\pi_2=0.5$.

**E-step.** With equal variances and weights, the probability depends only on $\exp(-(x-\mu)^2/8)$:

| $x$ | group 1 (center 0) | group 2 (center 10) | $r_{i1}$ | $r_{i2}$ |
|---|---|---|---|---|
| 1 | $e^{-1/8}=0.8825$ | $e^{-81/8}=0.00004$ | 0.99995 | 0.00005 |
| 2 | $e^{-4/8}=0.6065$ | $e^{-64/8}=0.00034$ | 0.99945 | 0.00055 |
| 5 | $e^{-25/8}=0.0439$ | $e^{-25/8}=0.0439$ | **0.5** | **0.5** |
| 8 | 0.00034 | 0.6065 | 0.00055 | 0.99945 |
| 9 | 0.00004 | 0.8825 | 0.00005 | 0.99995 |

The point in the middle ($x=5$) is exactly halfway: EM gives it 50% and 50%. A k-means would have had to choose.

**M-step for group 1.**
- $N_1 = 0.99995 + 0.99945 + 0.5 + 0.00055 + 0.00005 = 2.5$, so $\pi_1 = 2.5/5 = 0.5$.
- $\mu_1 = \dfrac{1\cdot0.99995 + 2\cdot0.99945 + 5\cdot0.5 + 8\cdot0.00055 + 9\cdot0.00005}{2.5} = \dfrac{5.5037}{2.5} \approx 2.20$
- $\sigma_1^2 = \dfrac{\sum_i r_{i1}(x_i-2.20)^2}{2.5} = \dfrac{5.42}{2.5} \approx 2.17$

By symmetry, $\mu_2 \approx 7.80$ and $\sigma_2^2 \approx 2.17$.

**Result after one iteration:** the centers moved closer to the data (from 0 and 10 to 2.2 and 7.8) and the bells narrowed (variance from 4 to 2.17). After a few iterations the centers settle near 1.5 and 8.5. That is all there is to it.

## 9. Practical details

- **Initialization:** chooses where the algorithm starts. Options: $K$ random points as means; or **k-means++** (more robust). Initial variances: the global variance of the data. Weights: $1/K$. With a fixed seed the results are reproducible, which is needed to compare the versions.
- **Local minima:** different starts can give different results. Several initializations can be tried, keeping the best one.
- **Numerical stability (log-sum-exp):** the probabilities $\pi_k\mathcal{N}$ can be tiny numbers (underflow). They are computed on a logarithmic scale and normalized by subtracting the maximum:
  $\log r_{ik} = \ell_{ik} - \left(m_i + \log\sum_j e^{\ell_{ij}-m_i}\right)$, with $\ell_{ik}=\log(\pi_k\mathcal{N}_{ik})$ and $m_i=\max_k \ell_{ik}$.
- **Regularization:** a variance that tends to 0 blows up the computation (a group that "locks onto" a single point). A small $\epsilon$ is added (for example $10^{-6}$): $\sigma^2 + \epsilon$ or $\Sigma + \epsilon I$.
- **Choice of $K$:** several values are tried and **BIC** or **AIC** are examined, which reward the fit to the data and penalize models that are too complex. The value with the lowest BIC is chosen.
- **Empty group:** if $N_k \approx 0$, the group is dead. It must be handled (reinitialize it or reduce $K$).

## 10. Pseudocode and data structures in C

```
read X (N x D), normalize
initialize pi[K], mu[K][D], var[K][D]            // diagonal
repeat until convergence:
    // E-step
    for each point i:
        for each group k:
            logp[k] = log(pi[k]) + log-gaussian(x[i], mu[k], var[k])
        r[i][k] = normalize with log-sum-exp
        ll += log-sum-exp(logp)                    // contribution to the log-likelihood
    // M-step
    for each group k:
        Nk = sum_i r[i][k]
        mu[k] = (sum_i r[i][k] * x[i]) / Nk
        var[k] = (sum_i r[i][k] * x[i]^2) / Nk - mu[k]^2  (+ epsilon)
        pi[k] = Nk / N
    check |ll - ll_prev| < tolerance
```

Arrays to allocate (all `double`):

| Array | Size | Notes |
|---|---|---|
| `X` | $N \times D$ | the data, the largest |
| `r` | $N \times K$ | responsibilities (can be avoided by fusing E-step and M-step) |
| `mu` | $K \times D$ | means |
| `var` | $K \times D$ (diagonal) or `Sigma` $K \times D \times D$ (full) | |
| `pi` | $K$ | weights |
| `Nk`, partial sums | $K$, $K \times D$ | M-step statistics |

Data memory: $N \cdot D \cdot 8$ bytes, for example 1 million points with 10 variables are 80 MB.

**Cost per iteration:** E-step $O(NKD)$ (diagonal) and M-step $O(NKD)$. With full covariance $O(NKD^2)$ plus $O(KD^3)$ for the inverses.

## 11. Where the parallelism is

- **E-step:** each point is **independent** of the others, so $X$ is split in blocks of rows among the MPI processes and, inside each one, among the OpenMP threads. No communication.
- **M-step:** these are **sums over the points**. Each process computes the sums over its block:
  $$N_k^{\text{loc}},\quad \sum_i r_{ik}x_i,\quad \sum_i r_{ik}x_i^2$$
  then a single `MPI_Allreduce` adds them across all processes. The amount communicated depends on $K$ and $D$ and **not on $N$**.
- **Parameters:** after the `Allreduce` every process has the same sums and recomputes $\mu_k$, $\sigma_k^2$, $\pi_k$ by itself (small computation, no broadcast).
- **Log-likelihood:** local sum plus an `MPI_Allreduce` of a single number, so that everyone decides the convergence in the same way.
- **Not parallelizable:** the iterations among themselves, because each uses the parameters of the previous one. This limits the speedup (Amdahl's law).

## 12. What to be able to say at the oral exam

1. **EM finds unlabeled groups** by modeling the data as a mixture of Gaussians.
2. **Two steps:** E (membership probability of each point to each group) and M (recomputation of center, shape and weight with weighted means).
3. **Why it is repeated:** knowing the groups is needed for the probabilities, knowing the probabilities is needed for the groups.
4. **What covariance does:** it describes how the variables vary together and gives the Gaussian its shape and tilt. We use the diagonal version for simplicity and cost $O(NKD)$.
5. **What is parallelized:** the points are independent, and the M-step consists of sums that are combined with `MPI_Allreduce`.
6. **Limits:** local minima (depends on the initialization), choice of $K$, sequential iterations.

## References

- A. P. Dempster, N. M. Laird, D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm", *Journal of the Royal Statistical Society B*, 1977 (the original paper).
- C. M. Bishop, *Pattern Recognition and Machine Learning*, chapter 9 (sections 9.2 and 9.3).
