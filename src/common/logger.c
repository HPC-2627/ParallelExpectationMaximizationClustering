/*
 * logger.c
 *
 * Writing of the experiments' .log file. The format of every block and the meaning of every key
 * are described in logger.h.
 */

#include "logger.h"

#include <math.h>      /* isnan */
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>  /* mkdir */
#include <sys/types.h>
#include <time.h>      /* time, gmtime_r, strftime */
#include <unistd.h>    /* gethostname */

/* Writes in 'out' (buffer of 'size' bytes) the model of the CPU, read from /proc/cpuinfo
 * (Linux-specific). If it cannot be read, it writes "unknown". */
static void cpu_model(char *out, size_t size)
{
    snprintf(out, size, "unknown"); /* fallback value */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f == NULL)
        return;
    char line[512];
    while (fgets(line, sizeof line, f) != NULL)
    {
        /* The wanted line has the form "model name\t: Intel(R) Core(TM) ..." */
        if (strncmp(line, "model name", 10) == 0)
        {
            char *colon = strchr(line, ':'); /* the value starts after the colon */
            if (colon != NULL)
            {
                colon++;
                while (*colon == ' ' || *colon == '\t') /* skip the leading spaces */
                    colon++;
                snprintf(out, size, "%s", colon);
                out[strcspn(out, "\r\n")] = '\0'; /* removes the trailing newline */
            }
            break; /* the first line is enough: the cores of a CPU have the same model */
        }
    }
    fclose(f);
}

/* Creates the folders of the path (like "mkdir -p" on the folder part of the path):
 * for "results/logs/x.log" it creates "results" and "results/logs". It works by temporarily
 * truncating a copy of the path at every '/', creating that folder and putting the '/' back.
 * Errors are ignored (mkdir fails if the folder already exists, which is normal): if a folder
 * cannot really be created, fopen() will fail afterwards and report it. */
static void make_parent_dirs(const char *path)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s", path);
    /* start from tmp + 1 so as not to try to create the root of an absolute path ("/") */
    for (char *p = tmp + 1; *p != '\0'; p++)
    {
        if (*p == '/')
        {
            *p = '\0';        /* truncate: tmp is now the path of the folder */
            mkdir(tmp, 0777); /* permissions 0777 modified by the user's umask */
            *p = '/';         /* restore */
        }
    }
}

int log_run(const char *path, const RunStats *s)
{
    make_parent_dirs(path);
    FILE *f = fopen(path, "a"); /* "a": appends at the end, previous runs are kept */
    if (f == NULL)
    {
        perror(path); /* prints "<path>: <reason of the error>" on stderr */
        return -1;
    }

    /* Node, CPU and date: needed to know where and when the measurement was made */
    char host[256] = "unknown";
    gethostname(host, sizeof host);
    host[sizeof host - 1] = '\0'; /* gethostname may not terminate the string if the name is too long */
    char cpu[256];
    cpu_model(cpu, sizeof cpu);

    /* Date and time in UTC (gmtime_r, the thread-safe version) in ISO 8601 format */
    char stamp[32];
    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

    /* One block of "key : value" lines per run. The key names are aligned to 15 characters
     * for readability. The times use %.9g (9 significant digits: enough for microseconds
     * without writing useless zeros). */
    fprintf(f, "==== run ====\n");
    fprintf(f, "timestamp      : %s\n", stamp);
    fprintf(f, "version        : %s\n", s->version);
    fprintf(f, "dataset        : %s\n", s->dataset);
    fprintf(f, "hostname       : %s\n", host);
    fprintf(f, "cpu_model      : %s\n", cpu);
    fprintf(f, "ranks          : %d\n", s->ranks);
    fprintf(f, "threads        : %d\n", s->threads);
    fprintf(f, "N              : %d\n", s->N);
    fprintf(f, "D              : %d\n", s->D);
    fprintf(f, "K              : %d\n", s->K);
    fprintf(f, "restarts       : %d\n", s->restarts);
    fprintf(f, "iterations     : %d\n", s->iterations);
    fprintf(f, "converged      : %d\n", s->converged);
    fprintf(f, "loglik         : %.12g\n", s->loglik);
    /* An unavailable ARI can be NaN or the error code -2.0 of eval.c (the real ARI is always
     * >= -1): in both cases "n/a" is written */
    if (isnan(s->ari) || s->ari < -1.5)
        fprintf(f, "ari            : n/a\n"); /* the dataset has no labels */
    else
        fprintf(f, "ari            : %.6f\n", s->ari);
    fprintf(f, "t_total        : %.9g\n", s->t_total);
    fprintf(f, "t_read         : %.9g\n", s->t_read);
    fprintf(f, "t_init         : %.9g\n", s->t_init);
    fprintf(f, "t_estep        : %.9g\n", s->t_estep);
    fprintf(f, "t_mstep        : %.9g\n", s->t_mstep);
    fprintf(f, "t_comm         : %.9g\n", s->t_comm);

    /* Time per iteration: defined only with a single run. With several restarts the times are
     * summed over all the runs, but the iterations are only those of the best one: the ratio
     * would make no sense. */
    if (s->restarts == 1 && s->iterations > 0)
        fprintf(f, "t_iteration    : %.9g\n", (s->t_estep + s->t_mstep + s->t_comm) / s->iterations);
    else
        fprintf(f, "t_iteration    : n/a\n");
    fprintf(f, "\n"); /* empty line: separates this block from the next one */

    fclose(f);
    return 0;
}
