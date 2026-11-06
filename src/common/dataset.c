/*
 * dataset.c
 *
 * Reading of the binary datasets and of the related labels. The file format is described in
 * dataset.h (8-byte header with N and D, then N*D doubles row by row).
 */

/* These two #define must come BEFORE any #include: they enable functions that plain standard
 * C does not offer.
 *   _POSIX_C_SOURCE 200809L : makes the POSIX functions (fseeko, fileno, access, ...) available
 *                             even when compiling with -std=c11 instead of gnu11.
 *   _FILE_OFFSET_BITS 64    : makes off_t / fseeko use 64 bits even on 32-bit systems, so that
 *                             files larger than 2 GB can be read (for example the synthetic
 *                             p = 16 of synth_rare: 1.6 GB, or brainweb). */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "dataset.h"
#include "common.h"

#include <errno.h>     /* errno: code of the last system error */
#include <limits.h>    /* INT_MAX */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>    /* strerror, strlen, strcmp */
#include <sys/stat.h>  /* fstat: size of the file */
#include <sys/types.h> /* off_t */
#include <unistd.h>    /* access */

#define HEADER_BYTES 8 /* the header is made of two int32 (N and D): 2 * 4 = 8 bytes */

/* Opens the file, reads the header and checks that the size is EXACTLY
 * HEADER_BYTES + N*D*8: it immediately detects a truncated file or a wrong format.
 *   path  path of the file
 *   fp    (output) the open file, positioned right AFTER the header; the caller closes it
 *   N, D  (output) rows and columns read from the header
 * Returns 0 if ok, -1 on error (message on stderr; the file is already closed). */
static int open_and_check(const char *path, FILE **fp, long *N, int *D)
{
    FILE *f = fopen(path, "rb"); /* "rb": read in binary mode */
    if (f == NULL)
    {
        fprintf(stderr, "dataset: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    int32_t h[2]; /* h[0] = N, h[1] = D */
    if (fread(h, sizeof(int32_t), 2, f) != 2)
    {
        fprintf(stderr, "dataset: %s: cannot read the header\n", path);
        fclose(f);
        return -1;
    }
    if (h[0] <= 0 || h[1] <= 0)
    {
        fprintf(stderr, "dataset: %s: invalid header (N=%d, D=%d)\n", path, h[0], h[1]);
        fclose(f);
        return -1;
    }
    struct stat st; /* st.st_size = size of the file in bytes */
    if (fstat(fileno(f), &st) != 0)
    {
        fprintf(stderr, "dataset: %s: cannot read the size: %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }
    /* The cast to off_t (64 bit) BEFORE the multiplication avoids int overflow:
     * with N = 250000 and D = 10 the product already exceeds one million and with big datasets
     * it would exceed 2^31 */
    off_t expected = (off_t)HEADER_BYTES + (off_t)h[0] * (off_t)h[1] * (off_t)sizeof(double);
    if (st.st_size != expected)
    {
        fprintf(stderr, "dataset: %s: wrong size (%lld bytes, expected %lld for N=%d, D=%d)\n",
                path, (long long)st.st_size, (long long)expected, h[0], h[1]);
        fclose(f);
        return -1;
    }
    *fp = f;
    *N = h[0];
    *D = h[1];
    return 0;
}

int dataset_read_header(const char *path, long *N, int *D)
{
    FILE *f = NULL;
    if (open_and_check(path, &f, N, D) != 0)
        return -1;
    fclose(f); /* only the header was needed */
    return 0;
}

/* Checks that the requested block [first_row, first_row + *n_rows) lies inside the file and
 * resolves the special value n_rows < 0 ("up to the end"), replacing it with the real number
 * of rows (this is why n_rows is passed by pointer).
 *   path    only for the error messages
 *   N_file  total rows of the file
 * Returns 0 if ok, -1 on error (message on stderr). */
static int resolve_block(const char *path, long N_file, long first_row, long *n_rows)
{
    if (first_row < 0 || first_row >= N_file)
    {
        fprintf(stderr, "dataset: %s: first_row=%ld out of range (the file has %ld rows)\n",
                path, first_row, N_file);
        return -1;
    }
    if (*n_rows < 0)
        *n_rows = N_file - first_row;
    if (*n_rows == 0 || first_row + *n_rows > N_file)
    {
        fprintf(stderr, "dataset: %s: block [%ld, %ld) out of range (the file has %ld rows)\n",
                path, first_row, first_row + *n_rows, N_file);
        return -1;
    }
    /* Dataset.N is an int: beyond INT_MAX rows the number would not fit */
    if (*n_rows > INT_MAX)
    {
        fprintf(stderr, "dataset: %s: too many rows (%ld) for an int\n", path, *n_rows);
        return -1;
    }
    return 0;
}

int dataset_load(const char *path, long first_row, long n_rows, Dataset *ds)
{
    FILE *f = NULL;
    long N_file;
    int D;
    if (open_and_check(path, &f, &N_file, &D) != 0)
        return -1;
    if (resolve_block(path, N_file, first_row, &n_rows) != 0)
    {
        fclose(f);
        return -1;
    }

    /* A single contiguous block of memory for all the values of the block of rows */
    size_t count = (size_t)n_rows * (size_t)D; /* values to read (not bytes) */
    double *X = (double *)malloc(count * sizeof(double));
    CHECK_ALLOC(X);

    /* The rows are contiguous in the file, so the block starts at byte
     * 8 + first_row * D * 8 and is read with a single fread. fseeko moves the read position
     * (the variant of fseek that accepts 64-bit offsets). */
    off_t offset = (off_t)HEADER_BYTES + (off_t)first_row * (off_t)D * (off_t)sizeof(double);
    if (fseeko(f, offset, SEEK_SET) != 0 || fread(X, sizeof(double), count, f) != count)
    {
        fprintf(stderr, "dataset: %s: error while reading the data\n", path);
        free(X);
        fclose(f);
        return -1;
    }
    fclose(f);

    ds->N = (int)n_rows; /* the check in resolve_block guarantees it fits in an int */
    ds->D = D;
    ds->first_row = first_row;
    ds->N_file = N_file;
    ds->X = X;
    return 0;
}

void dataset_free(Dataset *ds)
{
    free(ds->X);
    ds->X = NULL; /* a second free(NULL) is harmless: protects against double free */
    ds->N = ds->D = 0;
}

int dataset_labels_path(const char *data_path, char *out, size_t out_size)
{
    size_t len = strlen(data_path);
    /* The name must end with ".bin" (4 characters) and have at least one character before it */
    if (len < 5 || strcmp(data_path + len - 4, ".bin") != 0)
        return -1;
    /* The two naming conventions of the repository are tried: ".bin" is removed and the
     * suffix is appended. "%.*s" prints only the first (len - 4) characters of data_path. */
    static const char *suffixes[] = {".labels.bin", "_labels.bin"};
    for (int i = 0; i < 2; i++)
    {
        snprintf(out, out_size, "%.*s%s", (int)(len - 4), data_path, suffixes[i]);
        if (access(out, R_OK) == 0) /* the file exists and is readable */
            return 0;
    }
    return -1;
}

int dataset_load_labels(const char *data_path, long first_row, long n_rows, int32_t **labels)
{
    char path[4096];
    if (dataset_labels_path(data_path, path, sizeof path) != 0)
        return 1; /* no labels file: this is not an error */

    /* The total number of rows (N_file) is read from the header of the data file: the labels
     * file has no header and must contain exactly N_file int32 values */
    long N_file;
    int D;
    if (dataset_read_header(data_path, &N_file, &D) != 0)
        return -1;
    if (resolve_block(path, N_file, first_row, &n_rows) != 0)
        return -1;

    FILE *f = fopen(path, "rb");
    if (f == NULL)
    {
        fprintf(stderr, "dataset: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || st.st_size != (off_t)N_file * (off_t)sizeof(int32_t))
    {
        fprintf(stderr, "dataset: %s: wrong size (expected %ld int32 values)\n", path, N_file);
        fclose(f);
        return -1;
    }
    int32_t *L = (int32_t *)malloc((size_t)n_rows * sizeof(int32_t));
    CHECK_ALLOC(L);
    /* The label of row i is at byte i*4 (no header) */
    if (fseeko(f, (off_t)first_row * (off_t)sizeof(int32_t), SEEK_SET) != 0 ||
        fread(L, sizeof(int32_t), (size_t)n_rows, f) != (size_t)n_rows)
    {
        fprintf(stderr, "dataset: %s: error while reading the labels\n", path);
        free(L);
        fclose(f);
        return -1;
    }
    fclose(f);
    *labels = L; /* the caller must free() it */
    return 0;
}
