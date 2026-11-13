#!/bin/bash
# PBS job body: runs ONE version on ALL the datasets (real and synthetic) with ONE layout.
# It is not submitted by hand: src/script/pemc-submit.sh calls qsub on it, passing the layout
# (select, place, walltime) on the qsub command line and the variables below with -v.
# Submit from the root of the repository (the job starts in $PBS_O_WORKDIR). Build first: make all mpi-omp.
#
# Variables (-v):
#   VERSION         name of the version = bin/PEMC-<VERSION> (e.g. baseline-MPI-OpenMP-full)
#   NODES           nodes (PBS chunks)                        [MPI versions]
#   RANKS_PER_NODE  MPI processes per node (= mpiprocs)       [MPI versions]
#   THREADS         OpenMP threads per process                [MPI versions]
#   PLACE           pack | scatter | pack:excl | scatter:excl (only for the name of the log)
#   REPS            repetitions of every (dataset, layout)    [default 3]
#   SYNTH_P         sizes of the synthetic files, e.g. 01:04:16 [default all = 01:02:04:08:16]
#   SETS            real | synthetic | all                    [default all]
#   ONLY            only the datasets whose name contains this text (to test) [default: all]
#   SWEEP           none | K | D   [default none]. Runs for plot P1 (time per iteration vs K or D)
#                   instead of the normal dataset list; the logs go to results/logs/sweep/ so they
#                   are not mixed with the scaling runs:
#                     K  the dataset synth_clean_p$SWEEP_P with every K of K_LIST (-k changes, data is the same)
#                     D  the datasets synth_sweep_dDDD (D = D_LIST, K = SWEEP_K); generate them with
#                        generate-synthetic-dataset.py --sweep
#   K_LIST          values of K for SWEEP=K   [default "2 4 8 16 32 64"]
#   D_LIST          values of D for SWEEP=D   [default "5 10 20 35 50"]
#   SWEEP_P         size of synth_clean for the K sweep [default 01 = 250000 rows]
#   SWEEP_K         K used in the D sweep     [default 4]
#   REQUIRE_CPU     text that the CPU model of EVERY node must contain, else the job stops
#                   without writing logs [default 6418H; empty = no check]
#PBS -q shortCPUQ

VERSION=${VERSION:?VERSION is required}
NODES=${NODES:-1}
RANKS_PER_NODE=${RANKS_PER_NODE:-1}
THREADS=${THREADS:-1}
PLACE=${PLACE:-pack}
REPS=${REPS:-3}
SYNTH_P=${SYNTH_P:-all}
SETS=${SETS:-all}
ONLY=${ONLY:-}
SWEEP=${SWEEP:-none}
K_LIST=${K_LIST:-2 4 8 16 32 64}
D_LIST=${D_LIST:-5 10 20 35 50}
SWEEP_P=${SWEEP_P:-01}
SWEEP_K=${SWEEP_K:-4}

cd "$PBS_O_WORKDIR" || exit 1
EXE=./bin/PEMC-$VERSION
[ -x "$EXE" ] || { echo "error: $EXE not found (run: make all mpi-omp)" >&2; exit 1; }

# ---- CPU check --------------------------------------------------------------------------------
# The cluster mixes several Xeon generations: times measured on different CPUs cannot be compared.
# Every node of the job must have a CPU whose model name contains REQUIRE_CPU, otherwise the job
# stops BEFORE writing any log (no mixed data). REQUIRE_CPU= (empty) disables the check.
REQUIRE_CPU=${REQUIRE_CPU-6418H}
if [ -n "$REQUIRE_CPU" ]; then
    for h in $( { sort -u "$PBS_NODEFILE" 2>/dev/null || hostname; } ); do
        if [ "${h%%.*}" = "$(hostname -s)" ]; then
            m=$(grep -m1 "model name" /proc/cpuinfo)
        else
            m=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
                "$h" grep -m1 "'model name'" /proc/cpuinfo)
        fi
        if [[ "$m" != *"$REQUIRE_CPU"* ]]; then
            echo "error: node $h has CPU '${m#*: }', not $REQUIRE_CPU: job stopped, no log written" >&2
            exit 2
        fi
    done
    echo "CPU check: every node has $REQUIRE_CPU"
fi

# ---- datasets: name:path:K ------------------------------------------------------------------
# K = number of true classes (synthetic: clusters of the profile). parkinson has no classes: K=2 is arbitrary.
REAL="brainweb:12 cardiotocography:10 cardiovascular:2 cdc_diabetes:2 diabetes130:3 \
      eeg_eye_state:2 gene_rnaseq:5 heart_failure:2 parkinson:2 statlog_heart:2"
SYNTH="synth_clean:4 synth_overlap:5 synth_rare:6"
[ "$SYNTH_P" = all ] && SYNTH_P=01:02:04:08:16

LIST=()
if [ "$SETS" = real ] || [ "$SETS" = all ]; then
    for x in $REAL; do LIST+=("${x%%:*}:data/real/${x%%:*}/${x%%:*}.bin:${x##*:}"); done
fi
if [ "$SETS" = synthetic ] || [ "$SETS" = all ]; then
    for x in $SYNTH; do
        for p in ${SYNTH_P//:/ }; do
            LIST+=("${x%%:*}_p$p:data/synthetic/${x%%:*}/${x%%:*}_p$p.bin:${x##*:}")
        done
    done
fi

# Sweeps for plot P1 replace the list above (one dataset with several K, or several D with one K)
if [ "$SWEEP" = K ]; then
    LIST=()
    for k in $K_LIST; do
        LIST+=("ksweep_k$k:data/synthetic/synth_clean/synth_clean_p$SWEEP_P.bin:$k")
    done
elif [ "$SWEEP" = D ]; then
    LIST=()
    for d in $D_LIST; do
        n=$(printf "synth_sweep_d%03d" "$d")
        LIST+=("dsweep_d$d:data/synthetic/$n/${n}_p01.bin:$SWEEP_K")
    done
fi

# ---- how to launch ---------------------------------------------------------------------------
# One log file per job (the jobs run at the same time: a shared file would get mixed blocks).
# The placement is not stored in the log by the program, so it is in the file name.
case "$VERSION" in
    *sequential*)
        TAG=seq
        LAUNCH=("$EXE") ;;
    *)
        module load OpenMPI/4.1.5-GCC-12.3.0
        TAG=${PLACE//:/_}_${NODES}n${RANKS_PER_NODE}r${THREADS}t
        export OMP_NUM_THREADS=$THREADS
        # One thread per core, kept on the core of its rank (no migrations)
        export OMP_PROC_BIND=close
        export OMP_PLACES=cores
        # Binding: every rank gets THREADS cores (PE=THREADS). The ranks fill the slots of the
        # hostfile in order (map-by slot), i.e. they go where PBS put the chunks: with place=pack
        # PBS may put several chunks on the same node, so "ppr:R:node" (exactly R per node) fails.
        # Without it OpenMPI binds each rank to ONE core (default with np <= 2) and all its
        # OpenMP threads end up on that core (no speedup from the threads).
        # --report-bindings prints the cores of every rank in the .err file (to check it).
        # Nodes: OpenMPI does not see the other nodes of the job by itself (it sees only the first
        # one, and ppr:R:node then fails with more than one node): the list of the nodes assigned by
        # PBS is passed with --hostfile (as in Exercise 10). plm_rsh_args: ssh between the nodes
        # without host-key prompts. Without PBS (test outside the queue) the local node is used.
        HOSTS=()
        [ -n "${PBS_NODEFILE:-}" ] && HOSTS=(--hostfile "$PBS_NODEFILE"
            --mca plm_rsh_args "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR")
        # Multi-node runs go through the UCX wrapper (Ethernet interface of the cluster)
        LAUNCH=(mpiexec -n $((NODES * RANKS_PER_NODE)) "${HOSTS[@]}"
                --map-by slot:PE=${THREADS} --bind-to core --report-bindings
                ./src/script/mpi_wrapper.sh "$EXE") ;;
esac
# Sweep logs go in their own folder: the same dataset appears with several K, and plots-performance.py
# must not mix those runs with the scaling runs of results/logs/<model>/<stage>/
# Scaling logs: results/logs/<model>/<stage>/ with model = diagonal | full and stage = sequential |
# baseline | opt1 | opt2, taken from the version name (e.g. opt1-MPI-OpenMP-diagonal -> diagonal/opt1)
MODEL=${VERSION##*-}
case "$VERSION" in
    *sequential*) STAGE=sequential ;;
    *) STAGE=${VERSION%%-*} ;;
esac
LOGDIR=results/logs/$MODEL/$STAGE
[ "$SWEEP" != none ] && LOGDIR=results/logs/sweep
LOG=$LOGDIR/${VERSION}_${TAG}.log
mkdir -p "$LOGDIR"

echo "version $VERSION | layout $TAG | datasets ${#LIST[@]} | reps $REPS | log $LOG"
echo "nodes assigned by PBS:"; sort "$PBS_NODEFILE" 2>/dev/null | uniq -c

fail=0
for entry in "${LIST[@]}"; do
    name=${entry%%:*}; rest=${entry#*:}; path=${rest%%:*}; K=${rest##*:}
    [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && continue
    # Full covariance needs K*D^2 doubles per rank: gene_rnaseq (D = 20264) does not fit in memory
    if [[ "$VERSION" == *full ]] && [ "$name" = gene_rnaseq ]; then
        echo "skip $name (not feasible with full covariance)"; continue
    fi
    [ -f "$path" ] || { echo "skip $name: $path not found"; fail=$((fail + 1)); continue; }
    for ((rep = 1; rep <= REPS; rep++)); do
        echo "---- $name K=$K rep $rep/$REPS"
        "${LAUNCH[@]}" -d "$path" -k "$K" -l "$LOG" || { echo "FAILED: $name"; fail=$((fail + 1)); }
    done
done
echo "done, $fail failure(s)"
[ "$fail" -eq 0 ]
