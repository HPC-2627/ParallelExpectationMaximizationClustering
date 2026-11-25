#!/bin/bash
# Roofs of the roofline (plot P10): memory bandwidth of the node, MEASURED with STREAM, and peak
# GFLOP/s, CALCULATED (cores x GHz x FLOP/cycle). No EM run is needed: the points of P10 come from
# the logs already written (keys flops, bytes).
# Run from the root of the repository, on the login node of the cluster:
#
#   src/script/pemc-stream.sh submit    downloads stream.c (login node: the compute nodes may have
#                                       no internet) and submits ONE exclusive job on a Xeon 6252N
#                                       node (same CPU as all the runs) that compiles and runs it
#   src/script/pemc-stream.sh summary   bandwidth (Triad) and peak GFLOP/s for every thread count,
#                                       and the --peak-bw / --peak-gflops / --peak-gflops-compiled
#                                       to give to the plots
#
# The job body is this same script (qsub runs it with MODE=job).
# Output: results/logs/stream/stream_<node>.txt (raw STREAM output, one block per thread count).
# Save the output of summary next to it: src/script/pemc-stream.sh summary > results/logs/stream/stream_<node>-summary.txt
#
# What the job measures: STREAM with T OpenMP threads, T in THREADS_LIST, threads bound to cores
#   - close  : threads packed on consecutive cores (first socket first), like OMP_PROC_BIND=close
#              of the EM runs;
#   - spread : threads spread over the 4 sockets (all the memory controllers), only for 8 and 96.
# The 1-node points of P10 use 8 cores: the roofs to use are the ones of T = 8 (the full node,
# T = 96, is printed too).
#
# Peak GFLOP/s = T x GHZ x FLOP/cycle per core, in double precision. Cascade Lake (Gold 6252N) has
# 2 FP/FMA ports per core:
#   SSE2     (2 doubles, no FMA)           2 ports x 2 doubles        =  4 FLOP/cycle  <- the binaries
#                                                                                         (-O3, no -march)
#   AVX2+FMA (4 doubles, FMA = 2 FLOP)     2 ports x 4 doubles x 2    = 16 FLOP/cycle
#   AVX-512  (8 doubles, FMA = 2 FLOP)     2 ports x 8 doubles x 2    = 32 FLOP/cycle  <- machine peak
# GHZ is the base clock (2.3 GHz). With AVX-512 the cores run below the base clock and with few
# active cores the turbo goes above it: the peak is an upper bound, not a measure.
#
# Variables:
#   ARRAY_SIZE    elements per STREAM array [default 200000000: 3 arrays x 1.6 GB, far more than
#                 the 4 x 35.75 MB of L3, as STREAM requires (>= 4x the total cache)]
#   NTIMES        repetitions of every kernel, STREAM keeps the best [default 20]
#   THREADS_LIST  thread counts [default "1 2 4 8 16 24 48 96"]
#   GHZ           clock used for the peak [default 2.3]
#   NODE_FILTER / REQUIRE_CPU   as in pemc-submit.sh [default cpu_type=Xeon6252N / 6252N]
#PBS -q shortCPUQ

set -u
OUT=results/logs/stream
SRC=bin/stream/stream.c
URL=https://www.cs.virginia.edu/stream/FTP/Code/stream.c
ARRAY_SIZE=${ARRAY_SIZE:-200000000}
NTIMES=${NTIMES:-20}
THREADS_LIST=${THREADS_LIST:-1 2 4 8 16 24 48 96}
GHZ=${GHZ:-2.3}
NODE_FILTER=${NODE_FILTER-cpu_type=Xeon6252N}
REQUIRE_CPU=${REQUIRE_CPU-6252N}
NCPUS=96            # whole node (4 sockets x 24 cores): no other job shares its memory bandwidth
MEM=16gb

# peak <threads> <FLOP/cycle>: GFLOP/s
peak() { awk -v t="$1" -v f="$2" -v g="$GHZ" 'BEGIN { printf "%.1f", t * g * f }'; }

# ---- job body (on the compute node) ------------------------------------------------------------
if [ "${MODE:-}" = job ]; then
    cd "$PBS_O_WORKDIR" || exit 1
    m=$(grep -m1 "model name" /proc/cpuinfo)
    if [ -n "$REQUIRE_CPU" ] && [[ "$m" != *"$REQUIRE_CPU"* ]]; then
        echo "error: node $(hostname -s) has CPU '${m#*: }', not $REQUIRE_CPU: no output" >&2
        exit 2
    fi
    module purge
    module load OpenMPI/4.1.5-GCC-12.3.0      # same GCC 12.3.0 as the EM binaries
    # Compiled on the node itself: -march=native is the CPU of the node (STREAM wants the best
    # bandwidth, not the flags of EM). -mcmodel=medium: static arrays larger than 2 GB.
    EXE=bin/stream/stream-$(hostname -s)
    gcc -O3 -march=native -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE="$ARRAY_SIZE" \
        -DNTIMES="$NTIMES" "$SRC" -o "$EXE" || exit 1
    ncores=$(nproc)
    f=$OUT/stream_$(hostname -s).txt
    {
        echo "### node $(hostname -s)  CPU ${m#*: }  cores $ncores  $(date -Is)"
        echo "### ARRAY_SIZE $ARRAY_SIZE  NTIMES $NTIMES  gcc $(gcc -dumpfullversion)"
        export OMP_PLACES=cores
        for bind in close spread; do
            for t in $THREADS_LIST; do
                [ "$t" -le "$ncores" ] || continue
                [ "$bind" = spread ] && [ "$t" -ne 8 ] && [ "$t" -ne "$ncores" ] && continue
                echo "### threads $t bind $bind"
                OMP_NUM_THREADS=$t OMP_PROC_BIND=$bind "./$EXE"
            done
        done
        echo "### end"
    } > "$f.tmp" 2>&1 && mv "$f.tmp" "$f"
    echo "written $f"
    exit 0
fi

# ---- login node --------------------------------------------------------------------------------
case "${1:-}" in
submit)
    mkdir -p "$OUT" bin/stream results/logs/pbs
    if [ ! -s "$SRC" ]; then
        curl -fsSL "$URL" -o "$SRC" || { echo "error: download of $URL failed" >&2; exit 1; }
        echo "downloaded $SRC"
    fi
    select="1:ncpus=$NCPUS:mem=$MEM"
    [ -n "$NODE_FILTER" ] && select="$select:$NODE_FILTER"
    qsub -N stream -l "select=$select" -l place=pack:excl -l walltime=00:20:00 \
         -o "$PWD/results/logs/pbs/stream.out" -e "$PWD/results/logs/pbs/stream.err" \
         -v "MODE=job,ARRAY_SIZE=$ARRAY_SIZE,NTIMES=$NTIMES,THREADS_LIST=$THREADS_LIST,REQUIRE_CPU=$REQUIRE_CPU" \
         "$PWD/src/script/pemc-stream.sh"
    echo "when it is done: $0 summary"
    ;;
summary)
    ls "$OUT"/stream_*.txt >/dev/null 2>&1 || { echo "no output in $OUT (job not done?)" >&2; exit 1; }
    for f in "$OUT"/stream_*.txt; do
        [[ "$f" == *-summary.txt ]] && continue   # saved output of this command, not a STREAM run
        echo "== $f"
        head -1 "$f"
        printf "%-8s %-7s %12s | %10s %10s %10s\n" threads bind "Triad GB/s" "SSE2" "AVX2" "AVX-512"
        printf "%-8s %-7s %12s | %32s\n" "" "" "" "peak GFLOP/s at $GHZ GHz"
        # STREAM prints "Triad:  <best MB/s> <avg s> <min s> <max s>", MB = 1e6 bytes
        awk '/^### threads/ { t = $3; b = $5 } /^Triad:/ { print t, b, $2 / 1000 }' "$f" |
        while read -r t b bw; do
            printf "%-8s %-7s %12.1f | %10s %10s %10s\n" "$t" "$b" "$bw" \
                "$(peak "$t" 4)" "$(peak "$t" 16)" "$(peak "$t" 32)"
        done
        bw8=$(awk '/^### threads/ { t = $3; b = $5 } /^Triad:/ && t == 8 && b == "close" { printf "%.1f", $2 / 1000 }' "$f")
        echo
        echo "P10 (1-node points use 8 cores):"
        echo "  --peak-bw $bw8 --peak-gflops $(peak 8 32) --peak-gflops-compiled $(peak 8 4)"
        echo "  (peak-gflops = AVX-512 machine peak, peak-gflops-compiled = SSE2, -O3 without -march)"
        echo
    done
    ;;
*)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
