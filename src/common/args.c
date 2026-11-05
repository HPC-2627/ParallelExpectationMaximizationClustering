/*
 * args.c
 *
 * Reading of the command line (see args.h for the meaning of the fields of RunArgs).
 * It uses getopt_long() of the GNU library, which recognizes both short options (-k 4) and
 * long ones (--clusters 4).
 */

#include "args.h"

#include <getopt.h> /* getopt_long, optarg, struct option */
#include <stdio.h>
#include <stdlib.h> /* atoi, atol, atof, strtoull */

void args_defaults(RunArgs *a, const char *default_name)
{
    a->dataset = NULL; /* required: if it stays NULL, args_parse() reports the error */
    a->K = 0;          /* required: 0 = not given */
    a->n_rows = -1;    /* -1 = use all the rows of the file */
    a->max_iter = 200; /* enough for convergence on the datasets of the project */
    a->tol = 1e-6;     /* relative change of the log-likelihood below which it stops */
    a->seed = 42;
    a->restarts = 1; /* a single run: the measured time is that of a single EM */
    a->verbose = 0;
    a->log_path = NULL; /* the default path is decided in args_parse() */
    a->no_log = 0;
    a->name = default_name;
}

void args_usage(const char *prog)
{
    /* The message goes to stderr (not stdout) so it does not mix with the program output.
     * The six %s are the name of the executable, used in the "usage:" line and in the examples. */
    fprintf(stderr,
            "usage: %s -d FILE.bin -k K [options]\n"
            "\n"
            "  -d, --dataset FILE   dataset in binary format (header N, D + float64 row-major)\n"
            "  -k, --clusters K     number of clusters\n"
            "  -n, --rows N         use only the first N rows of the file (default: all)\n"
            "  -i, --max-iter N     maximum number of iterations (default: 200)\n"
            "  -t, --tol X          relative tolerance on the log-likelihood (default: 1e-6)\n"
            "  -s, --seed S         seed of the initialization (default: 42)\n"
            "  -r, --restarts R     number of initializations, the best one is kept (default: 1);\n"
            "                       the times are summed over all the restarts\n"
            "  -l, --log FILE.log   append a block with the results and the times to this log file\n"
            "                       (default: results/logs/<name>.log, created if missing)\n"
            "      --no-log         do not write the log\n"
            "      --name NAME      name of the version written in the log\n"
            "  -v, --verbose        print the log-likelihood of every iteration\n"
            "  -h, --help           this message\n"
            "\n"
            "single run:\n"
            "  %s -d data/synthetic/synth_clean/synth_clean_p01.bin -k 4\n"
            "\n"
            "Run from the root of the repository. K is the number of true classes of the dataset.\n"
            "\n"
            "all the synthetic datasets (K: synth_clean 4, synth_overlap 5, synth_rare 6;\n"
            "p01..p16 = size scaled by the number of processes of the scaling tests):\n"
            "  for p in 01 02 04 08 16; do\n"
            "    %s -d data/synthetic/synth_clean/synth_clean_p$p.bin -k 4\n"
            "    %s -d data/synthetic/synth_overlap/synth_overlap_p$p.bin -k 5\n"
            "    %s -d data/synthetic/synth_rare/synth_rare_p$p.bin -k 6\n"
            "  done\n"
            "\n"
            "all the real datasets (K = number of classes in the labels file):\n"
            "  for x in brainweb:12 cardiotocography:10 cardiovascular:2 cdc_diabetes:2 \\\n"
            "           diabetes130:3 eeg_eye_state:2 gene_rnaseq:5 heart_failure:2 \\\n"
            "           parkinson:2 statlog_heart:2; do\n"
            "    %s -d data/real/${x%%%%:*}/${x%%%%:*}.bin -k ${x##*:}\n"
            "  done\n"
            "  (parkinson has no labels, its continuous targets: K = 2 is an arbitrary choice)\n",
            prog, prog, prog, prog, prog, prog);
}

int args_parse(int argc, char **argv, RunArgs *a)
{
    /* Table of the long options. Every row is { long name, whether it takes an argument,
     * (flag, unused), code returned by getopt_long }.
     *   required_argument = the option needs a value (--rows 1000)
     *   no_argument       = the option is a switch (--verbose)
     * The returned code is the letter of the equivalent short option. The options --name and
     * --no-log have no short letter, so they use 'N' and 'L' as internal codes.
     * The last row of zeros ends the table (required by getopt_long). */
    static const struct option long_options[] = {
        {"dataset", required_argument, 0, 'd'},
        {"clusters", required_argument, 0, 'k'},
        {"rows", required_argument, 0, 'n'},
        {"max-iter", required_argument, 0, 'i'},
        {"tol", required_argument, 0, 't'},
        {"seed", required_argument, 0, 's'},
        {"restarts", required_argument, 0, 'r'},
        {"log", required_argument, 0, 'l'},
        {"name", required_argument, 0, 'N'},
        {"no-log", no_argument, 0, 'L'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    /* Loop over the options: getopt_long returns one at a time the code of the option found
     * (and puts its value in 'optarg'), then -1 when the options are finished.
     * The string "d:k:n:i:t:s:r:l:vh" lists the short options: a letter followed by ':'
     * takes a value, a lone letter (v, h) is a switch. */
    int c;
    while ((c = getopt_long(argc, argv, "d:k:n:i:t:s:r:l:vh", long_options, NULL)) != -1)
    {
        switch (c)
        {
        case 'd': a->dataset = optarg; break;               /* pointer inside argv: no copy */
        case 'k': a->K = atoi(optarg); break;               /* string -> int */
        case 'n': a->n_rows = atol(optarg); break;          /* string -> long */
        case 'i': a->max_iter = atoi(optarg); break;
        case 't': a->tol = atof(optarg); break;             /* string -> double */
        case 's': a->seed = strtoull(optarg, NULL, 10); break; /* string -> uint64, base 10 */
        case 'r': a->restarts = atoi(optarg); break;
        case 'l': a->log_path = optarg; break;
        case 'N': a->name = optarg; break;
        case 'L': a->no_log = 1; break;
        case 'v': a->verbose = 1; break;
        case 'h': args_usage(argv[0]); return 1; /* help requested: not an error */
        default: args_usage(argv[0]); return -1; /* unknown option (getopt already warned) */
        }
    }

    /* Validity checks. Note: atoi() returns 0 for a non-numeric string, so a wrong -k ends up
     * here as K < 1. */
    /* Dataset and number of clusters are required */
    if (a->dataset == NULL || a->K < 1)
    {
        fprintf(stderr, "error: -d (dataset) and -k (number of clusters, >= 1) are required\n\n");
        args_usage(argv[0]);
        return -1;
    }
    if (a->max_iter < 1 || a->restarts < 1 || a->tol < 0.0)
    {
        fprintf(stderr, "error: invalid value (max-iter >= 1, restarts >= 1, tol >= 0)\n");
        return -1;
    }

    /* Path of the log. By default the log is ALWAYS written, in results/logs/<name>.log:
     *   --no-log      -> no log (log_path = NULL)
     *   no option     -> default path, built in the internal buffer log_buf
     *   -l FILE       -> the path given by the user is kept */
    if (a->no_log)
        a->log_path = NULL;
    else if (a->log_path == NULL)
    {
        snprintf(a->log_buf, sizeof a->log_buf, "results/logs/%s.log", a->name);
        a->log_path = a->log_buf;
    }
    return 0;
}
