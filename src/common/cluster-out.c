/*
 * cluster-out.c
 *
 * Writing of the clustering result (labels, means, convergence curve). The format is described
 * in cluster-out.h.
 */

#include "cluster-out.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h> /* mkdir */
#include <sys/types.h>

/* Creates the folder 'path' and its parents, like "mkdir -p". Errors are ignored: if a folder
 * cannot be created, the fopen() that follows fails and reports it. */
static void make_dirs(const char *path)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p != '\0'; p++)
        if (*p == '/')
        {
            *p = '\0';
            mkdir(tmp, 0777); /* fails if it exists: normal */
            *p = '/';
        }
    mkdir(tmp, 0777); /* the last component (the path has no trailing '/') */
}

/* Writes in 'out' the name of the dataset file without folders and without the ".bin"
 * extension: "data/real/heart_failure/heart_failure.bin" -> "heart_failure". */
static void dataset_stem(const char *path, char *out, size_t size)
{
    const char *slash = strrchr(path, '/');
    const char *name = slash ? slash + 1 : path;
    snprintf(out, size, "%s", name);
    size_t n = strlen(out);
    if (n > 4 && strcmp(out + n - 4, ".bin") == 0)
        out[n - 4] = '\0';
}

int cluster_save(const char *dir, const char *dataset, const char *model, const GMMParams *p,
                 const int *labels, int N, const double *hist, int n_hist)
{
    char stem[256], base[4608], path[4700];
    dataset_stem(dataset, stem, sizeof stem);
    make_dirs(dir);
    snprintf(base, sizeof base, "%s/%s_%s_k%d", dir, stem, model, p->K);

    /* 1. labels: raw int32 array, N values */
    snprintf(path, sizeof path, "%s.labels.bin", base);
    FILE *f = fopen(path, "wb");
    if (f == NULL)
    {
        perror(path);
        return -1;
    }
    if (fwrite(labels, sizeof(int), (size_t)N, f) != (size_t)N)
    {
        perror(path);
        fclose(f);
        return -1;
    }
    fclose(f);

    /* 2. weights and means, one cluster per line (%.12g: enough digits for the plots) */
    snprintf(path, sizeof path, "%s.params.csv", base);
    f = fopen(path, "w");
    if (f == NULL)
    {
        perror(path);
        return -1;
    }
    fprintf(f, "cluster,pi");
    for (int d = 0; d < p->D; d++)
        fprintf(f, ",mu_%d", d);
    fprintf(f, "\n");
    for (int k = 0; k < p->K; k++)
    {
        fprintf(f, "%d,%.12g", k, p->pi[k]);
        for (int d = 0; d < p->D; d++)
            fprintf(f, ",%.12g", p->mu[(size_t)k * p->D + d]);
        fprintf(f, "\n");
    }
    fclose(f);

    /* 3. convergence curve */
    if (hist != NULL && n_hist > 0)
    {
        snprintf(path, sizeof path, "%s.loglik.csv", base);
        f = fopen(path, "w");
        if (f == NULL)
        {
            perror(path);
            return -1;
        }
        fprintf(f, "iteration,loglik\n");
        for (int i = 0; i < n_hist; i++)
            fprintf(f, "%d,%.12g\n", i, hist[i]);
        fclose(f);
    }
    return 0;
}
