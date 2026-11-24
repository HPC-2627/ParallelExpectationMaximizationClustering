#ifndef DATASET_H
#define DATASET_H

/*
 * dataset.h
 *
 * Reading of the datasets from binary files.
 *
 * ---------------------------------------------------------------------------------------
 * FILE FORMAT (the same for real and synthetic datasets)
 * ---------------------------------------------------------------------------------------
 *   bytes 0..3    int32  N    number of rows (points)
 *   bytes 4..7    int32  D    number of columns (variables)
 *   bytes 8..     N*D float64 (double) values, written ROW BY ROW:
 *                 row 0 (D values), row 1 (D values), ..., row N-1
 *   Expected total file size: 8 + N*D*8 bytes.
 *
 *   LABELS: if they exist, they are in a SEPARATE file with N int32 values, one per row
 *   (the true subgroup/class of that row; -1 = outlier, i.e. a point that does not belong to
 *   any group). They are used ONLY to evaluate the result (see eval.h): EM never sees them.
 *
 * ---------------------------------------------------------------------------------------
 * PARTIAL LOADING
 * ---------------------------------------------------------------------------------------
 * Only a block of rows [first_row, first_row + n_rows) can be loaded. In the MPI version every
 * rank will read only its own rows (rank r reads rows r*N0 to (r+1)*N0 - 1, where N0 = rows
 * per rank), so no process keeps the whole dataset in memory.
 * The functions of this file do not use MPI, so baseline-sequential uses them too.
 */

#include <stddef.h>
#include <stdint.h>

/* Data loaded in memory. */
typedef struct
{
    int N;          /* rows loaded in memory (may be less than N_file) */
    int D;          /* columns (variables) */
    long first_row; /* index, IN THE FILE, of the first loaded row (0 if loading from the start) */
    long N_file;    /* total rows present in the file (read from the header) */
    double *X;      /* the data: X[i*D + d] = variable d of row i (row-major order,
                       N*D values in a single contiguous block of memory) */
} Dataset;

/* Reads only the header (N and D) and checks that the file size is the expected one.
 * Writes the number of rows of the file in *N and the number of columns in *D.
 * Returns 0 if ok, -1 on error (the message goes to stderr). */
int dataset_read_header(const char *path, long *N, int *D);

/* Loads n_rows rows starting from row first_row (the first row of the file is row 0).
 * n_rows < 0 means "up to the end of the file".
 * Returns 0 if ok, -1 on error (message on stderr).
 * The memory (ds->X) is allocated here and is released with dataset_free(). */
int dataset_load(const char *path, long first_row, long n_rows, Dataset *ds);

/* Releases the memory of ds->X and zeroes N and D. It can be called once for every
 * successful dataset_load(). */
void dataset_free(Dataset *ds);

/* Finds the labels file corresponding to a data file. The repository has two naming
 * conventions, starting from "<name>.bin":
 *   "<name>.labels.bin"   (synthetic datasets)
 *   "<name>_labels.bin"   (real datasets)
 * The path found is written in 'out' (buffer of out_size bytes).
 * Returns 0 if the file exists, -1 otherwise (also if data_path does not end with ".bin"). */
int dataset_labels_path(const char *data_path, char *out, size_t out_size);

/* Loads the labels of rows [first_row, first_row + n_rows) (n_rows < 0 = up to the end).
 * Return values:
 *    0  loaded: the int32 array (allocated with malloc, to be released with free) is in *labels
 *    1  the dataset has no labels file (this is NOT an error: *labels is not touched)
 *   -1  error (wrong file size, failed read, ...) */
int dataset_load_labels(const char *data_path, long first_row, long n_rows, int32_t **labels);

#endif
