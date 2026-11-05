/*
 * rng.c
 *
 * SplitMix64 pseudorandom generator and derived functions (see rng.h for the reason of the
 * choice and for the meaning of the functions).
 */

#include "rng.h"

void rng_seed(Rng *r, uint64_t seed)
{
    r->state = seed;
}

/* SplitMix64 (Steele, Lea, Flood): small, fast and of good statistical quality.
 * At every call:
 *   1. the state advances by a fixed constant (0x9E37... = fractional part of the golden ratio
 *      multiplied by 2^64): this is the "sequence" that goes through all the 2^64 states;
 *   2. the state thus obtained is "scrambled" by two bit mixings (XOR with the same value
 *      shifted right) and two multiplications by constants: this turns a predictable sequence
 *      (state, state + c, state + 2c, ...) into numbers that look random.
 * The multiplications are modulo 2^64 (overflow is well defined for unsigned types). */
uint64_t rng_next(Rng *r)
{
    uint64_t z = (r->state += 0x9E3779B97F4A7C15ULL); /* advance the state and copy it to z */
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double rng_uniform(Rng *r)
{
    /* A double has 53 bits of mantissa. The 53 most significant bits of the 64-bit number are
     * taken (>> 11 discards the 11 least significant ones, which are the least "random"),
     * giving an integer in [0, 2^53), which is divided by 2^53 = 9007199254740992: the result
     * is in [0, 1). */
    return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

uint64_t rng_below(Rng *r, uint64_t n)
{
    /* remainder of the division: result in [0, n) */
    return rng_next(r) % n;
}

void rng_sample_distinct(Rng *r, uint64_t n, int k, uint64_t *out)
{
    /* Rejection sampling: a candidate is drawn and accepted only if it was not already drawn.
     * It is cheap because k (the number of clusters) is tiny compared to n (the number of
     * rows): collisions are rare and the check for duplicates costs O(k) per draw. If k were
     * close to n another method would be better (for example a partial Fisher-Yates shuffle). */
    int found = 0; /* how many distinct values I already have in out[0..found-1] */
    while (found < k)
    {
        uint64_t candidate = rng_below(r, n);
        int repeated = 0;
        for (int j = 0; j < found; j++)
        {
            if (out[j] == candidate)
            {
                repeated = 1;
                break;
            }
        }
        if (!repeated)
            out[found++] = candidate;
    }
}
