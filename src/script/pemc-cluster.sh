#!/bin/bash
# Runs the SEQUENTIAL BASELINE (diagonal and full) on the datasets of the clustering plots C1-C6
# and saves what src/plots-generate/plots-cluster.py needs. Run from the root of the repository,
# after "make" (sequential versions). Works both directly and with qsub (the PBS line below):
#   src/script/pemc-cluster.sh                 all the real datasets + synth_clean_p01
#   ONLY=heart src/script/pemc-cluster.sh      only the datasets whose name contains "heart"
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
#PBS -q shortCPUQ

KMIN=${KMIN:-2}
KMAX=${KMAX:-10}
ONLY=${ONLY:-}
SETS=${SETS:-all}
cd "${PBS_O_WORKDIR:-.}" || exit 1

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

fail=0
for model in diagonal full; do
    EXE=./bin/PEMC-baseline-sequential-$model
    [ -x "$EXE" ] || { echo "error: $EXE not found (run: make)" >&2; exit 1; }
    LOG=results/logs/cluster/baseline-sequential-$model.log
    for entry in "${LIST[@]}"; do
        name=${entry%%:*}; rest=${entry#*:}; path=${rest%%:*}; Kt=${rest##*:}
        [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && continue
        # Full covariance needs K*D^2 doubles: gene_rnaseq (D = 20531) does not fit in memory
        [ "$model" = full ] && [ "$name" = gene_rnaseq ] && { echo "skip $name full"; continue; }
        [ -f "$path" ] || { echo "skip $name: $path not found"; fail=$((fail + 1)); continue; }
        # K range, with the true K added if it is outside
        ks=$(seq "$KMIN" "$KMAX")
        [ "$Kt" -gt "$KMAX" ] && ks="$ks $Kt"
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
