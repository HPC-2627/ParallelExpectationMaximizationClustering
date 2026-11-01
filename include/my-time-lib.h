#ifndef MY_TIME_LIB_H
#define MY_TIME_LIB_H

#include <sys/time.h>
#include <math.h>
#include <stdio.h>

/**
 * Defines two timeval structures to store the start and end timestamps.
 * 'n' is the unique identifier for the timer instance.
 */
#define TIMER_DEF(n) struct timeval temp_1_##n = {0, 0}, temp_2_##n = {0, 0}

/**
 * Captures the current system time and stores it in the start variable.
 */
#define TIMER_START(n) gettimeofday(&temp_1_##n, (struct timezone *)0)

/**
 * Captures the current system time and stores it in the stop variable.
 */
#define TIMER_STOP(n) gettimeofday(&temp_2_##n, (struct timezone *)0)

/**
 * Calculates the elapsed time in microseconds (us).
 * Formula: (Seconds_diff * 1,000,000) + Microseconds_diff.
 */
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec - temp_1_##n.tv_sec) * 1.e6 + (temp_2_##n.tv_usec - temp_1_##n.tv_usec))

/**
 * Prints the elapsed time converted to seconds (s) to the standard output.
 * Uses a 'do-while(0)' block to ensure macro safety in conditional statements.
 * No trailing semicolon: the caller writes it (so it also works in 'if (x) TIMER_PRINT(t); else ...').
 * Do not call it inside a measured section: printing slows the program down.
 */
#define TIMER_PRINT(n)                                           \
    do                                                           \
    {                                                            \
        printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n) / 1e6); \
        fflush(stdout);                                          \
    } while (0)

/* --- Statistical Function Prototypes ---
 * A benchmark is repeated several times (after discarding the warm-up runs) and the
 * measured times are collected in an array 'v'. These functions summarize the array:
 * the value to report (mean or median) and its uncertainty (standard deviation),
 * which becomes the error bar in the timing plots.
 */

/**
 * Geometric mean of the array 'v' of length 'len'.
 * Use it to average ratios such as speedups over several datasets: it keeps a single
 * dataset with a huge speedup from dominating the result.
 */
double geometric_mean(const double *v, int len);

/**
 * Arithmetic mean of the array 'v' of length 'len'.
 * The typical value of stable measurements. It is pulled up by outliers
 * (e.g. a run disturbed by another job on the node), so it is also the input of std_dev().
 */
double arithmetic_mean(const double *v, int len);

/**
 * Median of the array 'v' of length 'len' (the array is not modified).
 * The typical value that is robust to outliers: a single slow run does not change it.
 * It is the preferred value to report for the execution times.
 */
double median(const double *v, int len);

/**
 * Sample standard deviation of the array 'v' of length 'len', given its mean 'mu'.
 * Measures how much the repeated runs differ from each other: it is the uncertainty
 * of the measurement and the size of the error bars. It divides by (len - 1) because
 * only a few repetitions (5-10) are available. If it is large compared with the mean,
 * the measurement is not reliable and the run must be repeated.
 */
double std_dev(const double *v, double mu, int len);

#endif
