#ifndef RNG_H
#define RNG_H

/*
 * rng.h
 *
 * Pseudorandom number generator (SplitMix64 algorithm).
 *
 * Why not use rand() of the standard library: the sequence of rand() depends on the
 * implementation of the C library, so two machines (or two compilers) could produce different
 * numbers with the same seed and start EM from different points.
 * With this generator the same seed ALWAYS gives the same numbers, everywhere: all the versions
 * (sequential, MPI baseline, optimizations) start from the same initialization. This is
 * essential to check that they obtain the same final log-likelihood.
 *
 * The generator is "pseudo"random: it is a deterministic function that, given the internal
 * state, produces the next number and updates the state.
 */

#include <stdint.h>

/* State of the generator: a single 64-bit number. */
typedef struct
{
    uint64_t state;
} Rng;

/* Sets the initial state from a seed. Same seed = same sequence. */
void rng_seed(Rng *r, uint64_t seed);

/* Next 64-bit random number (uniform over all the 2^64 values). */
uint64_t rng_next(Rng *r);

/* Random double in [0, 1) with 53 bits of randomness (upper end 1 excluded). */
double rng_uniform(Rng *r);

/* Random integer in [0, n) (n > 0). It uses the remainder of the division by n: the
 * distribution has a small bias (some values slightly more probable than others), negligible
 * because n is much smaller than 2^64 (here n is a number of rows). */
uint64_t rng_below(Rng *r, uint64_t n);

/* Draws k DISTINCT integers in [0, n) (requires k <= n) and writes them in out[0..k-1]. */
void rng_sample_distinct(Rng *r, uint64_t n, int k, uint64_t *out);

#endif
