#include <stdlib.h>
#include "my-time-lib.h"

/**
 * Calculates the Arithmetic Mean (Average).
 * Typical value of stable measurements; pulled up by outliers.
 */
double arithmetic_mean(const double *v, int len)
{
    if (len <= 0)
        return 0.0;
    double mu = 0.0;
    for (int i = 0; i < len; i++)
        mu += (double)v[i];
    return mu / (double)len;
}

/**
 * Calculates the Geometric Mean using logarithms to prevent numerical overflow.
 * Used to average ratios (e.g. speedups over several datasets).
 * Note ==>  Only values > 0 are meaningful: a value <= 0 adds 0 to the sum of the logs,
 *           so it is treated as 1 (and still counted in 'len').
 */
double geometric_mean(const double *v, int len)
{
    if (len <= 0)
        return 0.0;
    double log_sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        log_sum += (v[i] > 0) ? log((double)v[i]) : 0.0;           // Logarithm is only defined for positive numbers
    }
    return exp(log_sum / (double)len);
}

/**
 * Comparison function for qsort(): ascending order of doubles.
 */
static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

/**
 * Calculates the Median.
 * The central value of the sorted data: robust to outliers (e.g. a run slowed down by
 * another job on the node), so it is the value to report for the execution times.
 * The array is copied before sorting, so 'v' is left untouched.
 * With an even number of elements it returns the mean of the two central values.
 * Returns 0.0 if 'len' <= 0 or if the memory for the copy is not available.
 */
double median(const double *v, int len)
{
    if (len <= 0)
        return 0.0;
    double *copy = (double *)malloc((size_t)len * sizeof(double));
    if (copy == NULL)
        return 0.0;
    for (int i = 0; i < len; i++)
        copy[i] = v[i];
    qsort(copy, (size_t)len, sizeof(double), compare_doubles);
    double m = (len % 2 == 1) ? copy[len / 2] : 0.5 * (copy[len / 2 - 1] + copy[len / 2]);
    free(copy);
    return m;
}

/**
 * Calculates the Sample Standard Deviation (divides by len - 1).
 * It measures how much the repeated runs differ from each other: it is the uncertainty
 * of the measurement (error bar). The sample version is used because only a few
 * repetitions (5-10) are available.
 * v =  The array of values.
 * mu = The pre-calculated arithmetic mean.
 * len = The number of elements. With len <= 1 the deviation is not defined: returns 0.0.
 */
double std_dev(const double *v, double mu, int len)
{
    if (len <= 1)
        return 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < len; i++)
    {
        sum_sq += ((double)v[i] - mu) * ((double)v[i] - mu);           // Sum of squared differences from the mean
    }
    return sqrt(sum_sq / (double)(len - 1));        // Square root converts variance into standard deviation
}
