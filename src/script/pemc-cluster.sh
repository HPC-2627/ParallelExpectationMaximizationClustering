#!/bin/bash
# Runs the SEQUENTIAL BASELINE (diagonal and full) on the datasets of the clustering plots C1-C6
# and saves what src/plots-generate/plots-cluster.py needs. Run from the root of the repository,
# after "make" (sequential versions). Works both directly and with qsub (the PBS lines below):
#   qsub src/script/pemc-cluster.sh            everything, in parallel on one exclusive node (~30 min)
#   src/script/pemc-cluster.sh                 the same, in this shell
#   ONLY=heart src/script/pemc-cluster.sh      only the datasets whose name contains "heart", in ONE process
#
# Parallel: without ONLY and KS the script starts one process per dataset, and one per K for the
# slow datasets of SPLIT_K (default brainweb: alone it takes ~2 h), then waits for all of them.
# Every process is the same script with ONLY (and KS) set; its output is
# results/logs/cluster/run_<dataset>[_k<K>].txt and ends with "done, 0 failure(s)".
# PARALLEL=0 runs everything in one process, one dataset after the other (as before).
#
# What it produces
#   results/logs/cluster/<version>.log   one block per run, for EVERY K of the range (BIC/AIC vs K, C2/C6
#                                        and the summary table). A separate folder: the same dataset appears
#                                        with several K, and it must not mix with the scaling logs.
#   results/clusters/                    labels, means and log-likelihood curve (cluster-out.h), saved ONLY
#                                        for K = the number of true classes (C1, C3, C4, C5): saving every K
#                                        would write one labels file per K (brainweb: 47 MB each).
#
# Variables:  KMIN (default 2)  KMAX (default 10)   range of K for the BIC/AIC curves; the true K is always
#             included even if larger than KMAX.   ONLY   name filter.   SETS  real | synthetic | all (default all)
#             KS     explicit list of K, e.g. KS="8" or KS="2 3 4 5": replaces KMIN..KMAX and the true K is
#                    NOT added (to split a slow dataset over several processes, each with its own K)
#             SPLIT_K  datasets with one process per K [default brainweb].  PARALLEL=0  one process only
#
# PBS: one whole node (place=excl: no other job on it) with the same CPU as all the runs, 24 cores for
# the ~21 processes. The times are not used by C1-C6: the exclusive node is only to finish sooner.
#PBS -q shortCPUQ
#PBS -N pemc-cluster
#PBS -l select=1:ncpus=24:mem=48gb:cpu_type=Xeon6252N
#PBS -l place=excl
#PBS -l walltime=06:00:00
#PBS -j oe

KMIN=${KMIN:-2}
KMAX=${KMAX:-10}
KS=${KS:-}
ONLY=${ONLY:-}
SETS=${SETS:-all}
SPLIT_K=${SPLIT_K:-brainweb}
PARALLEL=${PARALLEL:-1}
cd "${PBS_O_WORKDIR:-.}" || exit 1
if [ -n "${PBS_JOBID:-}" ] && type module >/dev/null 2>&1; then
    module purge; module load OpenMPI/4.1.5-GCC-12.3.0   # same GCC as the build
fi

# name:path:K_true  (same list and K as pemc-job.sh; parkinson has no classes, K = 2 is arbitrary)
REAL="brainweb:12 cardiotocography:10 cardiovascular:2 cdc_diabetes:2 diabetes130:3 \
      eeg_eye_state:2 gene_rnaseq:5 heart_failure:2 parkinson:2 statlog_heart:2"
LIST=()
if [ "$SETS" = real ] || [ "$SETS" = all ]; then
    for x in $REAL; do LIST+=("${x%%:*}:data/real/${x%%:*}/${x%%:*}.bin:${x##*:}"); done
fi
# synthetic with known truth, small size (C3 shows found clusters next to the true classes)
if [ "$SETS" = synthetic ] || [ "$SETS" = all ]; then
    LIST+=("synth_clean_p01:data/synthetic/synth_clean/synth_clean_p01.bin:4"
           "synth_overlap_p01:data/synthetic/synth_overlap/synth_overlap_p01.bin:5")
fi

# ---- parallel: one process per dataset (per K for SPLIT_K), each one this script with ONLY/KS ----
if [ "$PARALLEL" != 0 ] && [ -z "$ONLY" ] && [ -z "$KS" ]; then
    mkdir -p results/logs/cluster
    echo "node $(hostname -s): $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
    pids=()
    for entry in "${LIST[@]}"; do
        name=${entry%%:*}; Kt=${entry##*:}
        if [[ " $SPLIT_K " == *" $name "* ]]; then
            ks=$(seq "$KMIN" "$KMAX"); [ "$Kt" -gt "$KMAX" ] && ks="$ks $Kt"
            for k in $ks; do
                PARALLEL=0 ONLY=$name KS=$k bash src/script/pemc-cluster.sh > "results/logs/cluster/run_${name}_k$k.txt" 2>&1 &
                pids+=($!); echo "started $name K=$k"
            done
        else
            PARALLEL=0 ONLY=$name bash src/script/pemc-cluster.sh > "results/logs/cluster/run_$name.txt" 2>&1 &
            pids+=($!); echo "started $name"
        fi
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
    echo "all done, $fail process(es) with failures (see results/logs/cluster/run_*.txt)"
    [ "$fail" -eq 0 ]
    exit
fi

fail=0
for model in diagonal full; do
    EXE=./bin/PEMC-baseline-sequential-$model
    [ -x "$EXE" ] || { echo "error: $EXE not found (run: make)" >&2; exit 1; }
    LOG=results/logs/cluster/baseline-sequential-$model.log
    for entry in "${LIST[@]}"; do
        name=${entry%%:*}; rest=${entry#*:}; path=${rest%%:*}; Kt=${rest##*:}
        [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && continue
        # Full covariance needs K*D^2 doubles: gene_rnaseq (D = 20264) does not fit in memory
        [ "$model" = full ] && [ "$name" = gene_rnaseq ] && { echo "skip $name full"; continue; }
        [ -f "$path" ] || { echo "skip $name: $path not found"; fail=$((fail + 1)); continue; }
        # K range, with the true K added if it is outside
        ks=$(seq "$KMIN" "$KMAX")
        [ "$Kt" -gt "$KMAX" ] && ks="$ks $Kt"
        [ -n "$KS" ] && ks=$KS
        for k in $ks; do
            echo "---- $model $name K=$k"
            args=(-d "$path" -k "$k" -w 0 -l "$LOG")       # -w 0: no warm-up, the times are not used here
            [ "$k" -eq "$Kt" ] && args+=(--save results/clusters)
            "$EXE" "${args[@]}" > /dev/null || { echo "FAILED: $model $name K=$k"; fail=$((fail + 1)); }
        done
    done
done
echo "done, $fail failure(s)"
[ "$fail" -eq 0 ]
