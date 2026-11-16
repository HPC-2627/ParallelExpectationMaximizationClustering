#!/bin/bash
# Submits to PBS the experiments of the 4 versions: baseline-sequential and baseline-MPI-OpenMP,
# diagonal and full. Run from the root of the repository, after "make all mpi-omp":
#   src/script/pemc-submit.sh [sequential | parallel | all]        (default: all)
#   src/script/pemc-submit.sh MODEL [STAGE]    MODEL = diagonal | full, STAGE = sequential | baseline | opt1 | opt2.
#                                           With a STAGE: only that version of that model, on ALL the datasets
#                                           (one stage at a time keeps the number of jobs under the per-user
#                                           limit of the queue). Without STAGE: every written stage of the model.
#   src/script/pemc-submit.sh sweep                                 sweeps of K and D for plot P1
#   DRY_RUN=1 src/script/pemc-submit.sh ...  only prints the qsub commands
#   PLACES_LIST="pack scatter" src/script/pemc-submit.sh diagonal baseline   only these placements (no :excl)
#   BLOCK=64 src/script/pemc-submit.sh full opt1      rows per block of opt1 full (sweep over B, logs
#                                           in results/logs/block/)
#   SKIP=4 src/script/pemc-submit.sh diagonal opt1   does NOT submit the first 4 jobs (same order as the
#                                           normal run): to send the rest after "per-user limit" errors
#
# One job = one version x one layout; it runs ALL the datasets (pemc-job.sh), so the number of
# jobs is small. Layout of a parallel job: NODES chunks, each with RANKS_PER_NODE MPI processes
# and THREADS OpenMP threads per process => ncpus per chunk = RANKS_PER_NODE x THREADS.
# The resources are given to qsub on the command line (they override any #PBS line).
#
# Placement (-l place=...):
#   pack          the chunks go on as FEW nodes as possible (if one node has enough cores, the
#                 "multi-node" job runs on a single node!)
#   scatter       every chunk on a DIFFERENT node
#   <x>:excl      same, but the nodes are used by this job only (no other job disturbs the timing)
# With NODES=1 pack and scatter are the same thing: only pack and pack:excl are submitted.

# ---- configuration (edit here) ----------------------------------------------------------------
# Layouts of the parallel versions: NODES:RANKS_PER_NODE:THREADS
CONFIGS=(
    # one node, 8 cores: threads vs ranks at a fixed number of cores
    1:1:8  1:2:4  1:4:2  1:8:1
    # several nodes, 8 cores each: 16 and 32 cores
    2:2:4  4:2:4
)
# CONFIGS_LIST overrides them, e.g. CONFIGS_LIST="2:2:4 4:2:4" (only the multi-node layouts)
[ -n "${CONFIGS_LIST:-}" ] && CONFIGS=($CONFIGS_LIST)
# Placements. PLACES_LIST overrides them, e.g. PLACES_LIST="pack scatter" (no :excl: on a busy cluster
# no node is ever completely empty, so the :excl jobs stay queued for a very long time)
PLACES=(${PLACES_LIST:-pack scatter pack:excl scatter:excl})
MEM=8gb                  # per chunk (per node)
WALLTIME=06:00:00        # per job: it runs every dataset; check the limit of shortCPUQ
REPS=${REPS:-3}          # repetitions of every (dataset, layout); override: REPS=1 src/script/pemc-submit.sh ...
SYNTH_P=all              # sizes of the synthetic files: all | e.g. 01:16
SETS=all                 # real | synthetic | all
# Node type: extra PBS resources added to EVERY chunk of the select, so that all the jobs (all the
# versions) run on the same kind of CPU and the times can be compared. The cluster mixes several
# Xeon generations (E5-2650 v3 ... Gold 6418H): without it every job lands on a different one.
# PBS resource cpu_type (see "pbsnodes -a"): Xeon6252N = 87 nodes (the most common type, short
# queues, enough nodes for the 4-node scatter jobs). Others: Xeon5218 (29), Xeon5118 (25),
# Xeon6140M (22), Xeon2680 (10), Xeon6418H (7). NODE_FILTER= (empty) = any node.
NODE_FILTER=${NODE_FILTER-cpu_type=Xeon6252N}
# CPU check inside the job (pemc-job.sh): every node must have this CPU model, otherwise the job
# stops without writing logs (safety net for NODE_FILTER: it must name the same CPU). Empty = no check.
REQUIRE_CPU=${REQUIRE_CPU-6252N}
# -----------------------------------------------------------------------------------------------

# Versions that are WRITTEN (one source file with the code, built by the makefile). Add the new ones
# here when they are ready (and to MPI_BINS in the makefile). Not written yet: opt2-MPI-OpenMP-full.
WRITTEN=" baseline-sequential-diagonal baseline-MPI-OpenMP-diagonal opt1-MPI-OpenMP-diagonal
          opt2-MPI-OpenMP-diagonal baseline-sequential-full baseline-MPI-OpenMP-full
          opt1-MPI-OpenMP-full "

WHAT=${1:-all}
LOGDIR=$PWD/results/logs/pbs
mkdir -p "$LOGDIR"

SKIP=${SKIP:-0}          # number of jobs, in submission order, not to submit (already submitted)
NJOB=0
submit() {  # submit NAME SELECT PLACE VARS...
    local name=$1 select=$2 place=$3; shift 3
    [ -n "${BLOCK:-}" ] && name="$name-b$BLOCK"   # one .out/.err per block size
    NJOB=$((NJOB + 1))
    [ "$NJOB" -le "$SKIP" ] && return 0
    [ -n "$NODE_FILTER" ] && select="$select:$NODE_FILTER"
    local vars; vars=$(IFS=,; echo "$*")
    local cmd=(qsub -N "$name" -l "select=$select" -l "place=$place" -l "walltime=$WALLTIME"
               -o "$LOGDIR/$name.out" -e "$LOGDIR/$name.err" -v "$vars" src/script/pemc-job.sh)
    if [ -n "$DRY_RUN" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

short() { case $1 in baseline-sequential-diagonal) echo sd;; baseline-sequential-full) echo sf;;
                     baseline-MPI-OpenMP-diagonal) echo bd;; baseline-MPI-OpenMP-full) echo bf;;
                     opt1-MPI-OpenMP-diagonal) echo o1d;; opt2-MPI-OpenMP-diagonal) echo o2d;;
                     opt1-MPI-OpenMP-full) echo o1f;; opt2-MPI-OpenMP-full) echo o2f;; esac; }
pl() { case $1 in pack) echo pk;; scatter) echo sc;; pack:excl) echo pkx;; scatter:excl) echo scx;; esac; }

# Sweeps (plot P1): time per iteration vs K and vs D, ONE layout (1 node, 1 rank x 8 threads,
# node used only by this job) so the only thing that changes is K or D. The D sweep needs the
# files made by: python3 src/dataset-managed/generate-synthetic-dataset.py --sweep
# The logs go to results/logs/sweep/ (see pemc-job.sh).
SWEEP_LAYOUT=1:1:8
if [ "$WHAT" = sweep ]; then
    for sw in K D; do
        for v in baseline-sequential-diagonal baseline-sequential-full; do
            submit "$(short $v)-sw$sw" "1:ncpus=1:mem=$MEM" pack:excl "VERSION=$v,SWEEP=$sw,REPS=$REPS,REQUIRE_CPU=$REQUIRE_CPU"
        done
        for v in baseline-MPI-OpenMP-diagonal baseline-MPI-OpenMP-full; do
            IFS=: read -r N R T <<< "$SWEEP_LAYOUT"
            submit "$(short $v)-sw$sw" "$N:ncpus=$((R * T)):mpiprocs=$R:mem=$MEM" pack:excl \
                   "VERSION=$v,SWEEP=$sw,NODES=$N,RANKS_PER_NODE=$R,THREADS=$T,PLACE=pack:excl,REPS=$REPS,REQUIRE_CPU=$REQUIRE_CPU"
        done
    done
    exit 0
fi

COMMON="REPS=$REPS,SYNTH_P=$SYNTH_P,SETS=$SETS,REQUIRE_CPU=$REQUIRE_CPU"
# Block size of opt1 full (sweep over B): passed to the jobs only if set; the logs then go to
# results/logs/block/ (see pemc-job.sh). Example, 1 node only, a few datasets:
#   BLOCK=64 CONFIGS_LIST="1:1:8" PLACES_LIST="pack:excl" src/script/pemc-submit.sh full opt1
[ -n "${BLOCK:-}" ] && COMMON="$COMMON,BLOCK=$BLOCK"

# Parallel layouts of one version: every layout of CONFIGS x every placement
submit_parallel() {  # submit_parallel VERSION
    local v=$1 c p N R T
    for c in "${CONFIGS[@]}"; do
        IFS=: read -r N R T <<< "$c"
        for p in "${PLACES[@]}"; do
            [ "$N" -eq 1 ] && [[ "$p" == scatter* ]] && continue
            submit "$(short $v)-$(pl $p)-${N}x${R}x${T}" \
                   "$N:ncpus=$((R * T)):mpiprocs=$R:mem=$MEM" "$p" \
                   "VERSION=$v,NODES=$N,RANKS_PER_NODE=$R,THREADS=$T,PLACE=$p,$COMMON"
        done
    done
}

# One model (diagonal | full), one stage or all the written stages, all the datasets
if [ "$WHAT" = diagonal ] || [ "$WHAT" = full ]; then
    MODEL=$WHAT
    case "$2" in
        "")         STAGES="sequential baseline opt1 opt2" ;;
        sequential|baseline|opt1|opt2) STAGES=$2 ;;
        *) echo "unknown stage '$2' (sequential | baseline | opt1 | opt2)" >&2; exit 1 ;;
    esac
    for st in $STAGES; do
        case $st in
            sequential) v=baseline-sequential-$MODEL ;;
            baseline)   v=baseline-MPI-OpenMP-$MODEL ;;
            *)          v=$st-MPI-OpenMP-$MODEL ;;
        esac
        if [[ " $(echo $WRITTEN) " != *" $v "* ]]; then
            [ -n "$2" ] && { echo "$v is not written yet (add it to WRITTEN here and to MPI_BINS in the makefile)" >&2; exit 1; }
            continue   # no stage given: the versions not written are skipped
        fi
        if [ "$st" = sequential ]; then
            submit "$(short $v)-seq" "1:ncpus=1:mem=$MEM" pack:excl "VERSION=$v,$COMMON"
        else
            submit_parallel "$v"
        fi
    done
    exit 0
fi

if [ "$WHAT" = sequential ] || [ "$WHAT" = all ]; then
    for v in baseline-sequential-diagonal baseline-sequential-full; do
        # 1 core, node used only by this job (clean serial time)
        submit "$(short $v)-seq" "1:ncpus=1:mem=$MEM" pack:excl "VERSION=$v,$COMMON"
    done
fi

if [ "$WHAT" = parallel ] || [ "$WHAT" = all ]; then
    for v in baseline-MPI-OpenMP-diagonal baseline-MPI-OpenMP-full; do
        submit_parallel "$v"
    done
fi
