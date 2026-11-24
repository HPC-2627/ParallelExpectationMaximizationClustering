# EM clustering explained
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

### Covariance
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
| Cost per iteration | $O(NKD^2)$ + factorizations $O(KD^3)$ | $O(NKD)$ |
| Risk | unstable with high $D$ and few data | assumes independent variables inside the group |

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
Starting values are needed for $\pi_k$, $\mu_k$, $\Sigma_k$. A simple choice, valid because the data are standardized (mean 0, variance 1):
- $\mu_k$ = $K$ distinct points $x_i$ chosen at random;
- $\Sigma_k = I$ (identity), that is $\sigma^2_{kd} = 1$ in the diagonal case;
- $\pi_k = 1/K$ (all groups weigh the same).

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

The covariance uses the **new** mean $\mu_k$ just computed, so the mean is updated first.

A point with $r_{ik}=0.9$ counts "almost fully" in group $k$. One with $r_{ik}=0.01$ almost does not count.

### Convergence check
$\mathcal{L}$ is computed and the algorithm stops when the **relative** change is below a tolerance $\varepsilon$ (for example $10^{-6}$):
$$\frac{|\mathcal{L}_{\text{new}}-\mathcal{L}_{\text{old}}|}{|\mathcal{L}_{\text{new}}|} \le \varepsilon$$
Otherwise it goes back to the E-step. Being relative, the threshold does not depend on the scale of $\mathcal{L}$. A **maximum number of iterations** is also fixed, as a safety net in case convergence is very slow.

### Final result
At the end every point has $K$ probabilities $r_{ik}$. To get a "hard" label, the most probable group is taken:
$$\hat{z}_i = \arg\max_k \, r_{ik}$$
If the dataset has true labels, they can be compared with $\hat{z}_i$ (for example with the Adjusted Rand Index).

## 8. Practical details

### Working with logarithms (log-sum-exp)
With many variables the densities $\mathcal{N}(x_i \mid \mu_k, \Sigma_k)$ are extremely small (the log is hundreds of units below zero), and computed directly they become $0$ (underflow). So everything is done with logarithms: let $a_k = \log\pi_k + \log\mathcal{N}(x_i \mid \mu_k, \Sigma_k)$ and $m = \max_k a_k$. Then
$$\log\sum_{k} e^{a_k} = m + \log\sum_{k} e^{a_k - m}, \qquad r_{ik} = e^{\,a_k - \log\sum_j e^{a_j}}$$
The largest term becomes $e^0 = 1$, so the sum is never $0$ nor infinite. The same quantity gives the contribution of point $i$ to $\mathcal{L}$.

### Degenerate groups
- If a group captures very few (or identical) points, its variance goes to $0$: the density contains $1/\sigma^2$ and $\log\sigma^2$, so $\mathcal{L}$ would become infinite. A small minimum is imposed: $\sigma^2_{kd} \ge 10^{-6}$ (in the full case a small value is added to the diagonal of $\Sigma_k$).
- If a group stays empty ($N_k \approx 0$), its mean and variance cannot be computed (division by $\sim 0$): they are kept as they were, and $\pi_k$ gets a tiny positive value so that $\log\pi_k$ is not $-\infty$.

### Full covariance without the inverse
$\Sigma_k^{-1}$ and $\det\Sigma_k$ are obtained from the **Cholesky factorization** $\Sigma_k = L_k L_k^T$ ($L_k$ lower triangular):
$$\log\det\Sigma_k = 2\sum_{d}\log (L_k)_{dd}, \qquad (x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) = \|y\|^2 \ \text{ with } \ L_k\, y = x-\mu_k$$
Solving the triangular system is cheaper and more stable than computing the inverse.

### Local maxima and restarts
EM converges only to a **local** maximum, which depends on the initialization. The usual remedy is to run it from several random starts and keep the one with the highest $\mathcal{L}$.