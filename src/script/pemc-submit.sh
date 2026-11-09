#!/bin/bash
# Submits to PBS the experiments of the 4 versions: baseline-sequential and baseline-MPI-OpenMP,
# diagonal and full. Run from the root of the repository, after "make all mpi-omp":
#   src/script/pemc-submit.sh [sequential | parallel | all]        (default: all)
#   DRY_RUN=1 src/script/pemc-submit.sh      only prints the qsub commands
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
PLACES=(pack scatter pack:excl scatter:excl)
MEM=8gb                  # per chunk (per node)
WALLTIME=06:00:00        # per job: it runs every dataset; check the limit of shortCPUQ
REPS=1                   # repetitions of every (dataset, layout)
SYNTH_P=all              # sizes of the synthetic files: all | e.g. 01:16
SETS=all                 # real | synthetic | all
# -----------------------------------------------------------------------------------------------

WHAT=${1:-all}
LOGDIR=$PWD/results/logs/pbs
mkdir -p "$LOGDIR"

submit() {  # submit NAME SELECT PLACE VARS...
    local name=$1 select=$2 place=$3; shift 3
    local vars; vars=$(IFS=,; echo "$*")
    local cmd=(qsub -N "$name" -l "select=$select" -l "place=$place" -l "walltime=$WALLTIME"
               -o "$LOGDIR/$name.out" -e "$LOGDIR/$name.err" -v "$vars" src/script/pemc-job.sh)
    if [ -n "$DRY_RUN" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

short() { case $1 in baseline-sequential-diagonal) echo sd;; baseline-sequential-full) echo sf;;
                     baseline-MPI-OpenMP-diagonal) echo bd;; baseline-MPI-OpenMP-full) echo bf;; esac; }
pl() { case $1 in pack) echo pk;; scatter) echo sc;; pack:excl) echo pkx;; scatter:excl) echo scx;; esac; }

COMMON="REPS=$REPS,SYNTH_P=$SYNTH_P,SETS=$SETS"

if [ "$WHAT" = sequential ] || [ "$WHAT" = all ]; then
    for v in baseline-sequential-diagonal baseline-sequential-full; do
        # 1 core, node used only by this job (clean serial time)
        submit "$(short $v)-seq" "1:ncpus=1:mem=$MEM" pack:excl "VERSION=$v,$COMMON"
    done
fi

if [ "$WHAT" = parallel ] || [ "$WHAT" = all ]; then
    for v in baseline-MPI-OpenMP-diagonal baseline-MPI-OpenMP-full; do
        for c in "${CONFIGS[@]}"; do
            IFS=: read -r N R T <<< "$c"
            for p in "${PLACES[@]}"; do
                [ "$N" -eq 1 ] && [[ "$p" == scatter* ]] && continue
                submit "$(short $v)-$(pl $p)-${N}x${R}x${T}" \
                       "$N:ncpus=$((R * T)):mpiprocs=$R:mem=$MEM" "$p" \
                       "VERSION=$v,NODES=$N,RANKS_PER_NODE=$R,THREADS=$T,PLACE=$p,$COMMON"
            done
        done
    done
fi
