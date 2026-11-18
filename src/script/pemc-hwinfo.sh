#!/bin/bash
# Hardware of ALL the nodes used by the experiments (table "Hardware Infrastructure" of the report).
# Run from the root of the repository, on the login node of the cluster:
#
#   src/script/pemc-hwinfo.sh submit     one small PBS job per node (body: pemc-hwprobe.sh), only
#                                        for the nodes that have no valid output yet and no job in
#                                        the queue; if the queue is full ("per-user limit") it waits
#                                        30 s and tries again, so it can be launched once for all
#   src/script/pemc-hwinfo.sh status     done / in the queue (with the reason) / missing, per node
#   src/script/pemc-hwinfo.sh software   compiler, MPI and UCX versions -> results/hwinfo/software.txt
#   src/script/pemc-hwinfo.sh summary    what is the same on all the nodes and what changes
#                                        (RAM, DIMMs, NIC, kernel per node; DIMMs per group)
#
# Then copy results/hwinfo/ to the laptop (scp -r, see doc/exec-cluster.md).
#
# The nodes are the 40 of the list NODES below: all the nodes used by the runs of both team members,
# taken from their PBS outputs (results/logs/pbs/*.out). NODES_FILE=<file> uses another list.
# The raw outputs in results/hwinfo/ are not versioned: the summary is results/tables/hardware.md.
# A node is "done" when results/hwinfo/<node>.txt contains the line MemTotal (job completed). A node
# can stay in the queue for a long time if other users' jobs fill it ("Insufficient amount of
# resource: host"): PBS gives no estimate, the job starts when the node is free.

set -u
OUT=results/hwinfo
PROBE=$PWD/src/script/pemc-hwprobe.sh

# Nodes used by the experiments (1 in c10, 18 in c11, 21 in c12)
NODES="
    hpc3-c10-n01 hpc3-c11-n01 hpc3-c11-n02 hpc3-c11-n03 hpc3-c11-n04 hpc3-c11-n05 hpc3-c11-n06
    hpc3-c11-n10 hpc3-c11-n13 hpc3-c11-n19 hpc3-c11-n22 hpc3-c11-n23 hpc3-c11-n24 hpc3-c11-n25
    hpc3-c11-n26 hpc3-c11-n27 hpc3-c11-n28 hpc3-c11-n29 hpc3-c11-n30 hpc3-c12-n08 hpc3-c12-n19
    hpc3-c12-n20 hpc3-c12-n22 hpc3-c12-n23 hpc3-c12-n24 hpc3-c12-n25 hpc3-c12-n26 hpc3-c12-n27
    hpc3-c12-n28 hpc3-c12-n30 hpc3-c12-n31 hpc3-c12-n32 hpc3-c12-n33 hpc3-c12-n40 hpc3-c12-n41
    hpc3-c12-n42 hpc3-c12-n43 hpc3-c12-n46 hpc3-c12-n47 hpc3-c12-n51
"

mkdir -p "$OUT"
NODES_FILE=${NODES_FILE:-$OUT/nodes.txt}
[ -s "$NODES_FILE" ] || echo $NODES | tr ' ' '\n' > "$NODES_FILE"

# done <node>: the job of the node has completed and written its data
done_node() { grep -q "^MemTotal" "$OUT/$1.txt" 2>/dev/null; }
# queued <node>: id of the job hw-<node> of this user still in PBS (queued or running), or empty
queued() { qselect -u "$USER" -N "hw-$1" 2>/dev/null | head -1; }

case "${1:-}" in
submit)
    echo "nodes: $(wc -l < "$NODES_FILE")  ($(cut -d- -f2 "$NODES_FILE" | sort | uniq -c | xargs))"
    for n in $(cat "$NODES_FILE"); do
        if done_node "$n"; then continue; fi
        if [ -n "$(queued "$n")" ]; then echo "$n: already in the queue"; continue; fi
        rm -f "$OUT/$n.txt"   # output of a failed job: written again by the new one
        # 1 core, 1 GB, 2 minutes on exactly that node; stdout+stderr in results/hwinfo/<node>.txt.
        # qsub fails when the per-user limit of the queue is reached: wait and try again.
        until qsub -q shortCPUQ -N "hw-$n" -l select=1:ncpus=1:mem=1gb:host="$n" -l walltime=00:02:00 \
                   -j oe -o "$PWD/$OUT/$n.txt" "$PROBE"; do
            echo "$n: queue full, new attempt in 30 s"
            sleep 30
        done
    done
    echo "check with: $0 status"
    ;;
status)
    nd=0; nq=0; nm=0
    for n in $(cat "$NODES_FILE"); do
        if done_node "$n"; then
            nd=$((nd + 1))
        elif j=$(queued "$n"); [ -n "$j" ]; then
            nq=$((nq + 1))
            echo "QUEUED  $n  $(qstat -f "$j" | grep -E "job_state|comment" | xargs)"
        else
            nm=$((nm + 1))
            echo "MISSING $n  (run: $0 submit)"
        fi
    done
    echo "done $nd, queued $nq, missing $nm (of $(wc -l < "$NODES_FILE"))"
    ;;
software)
    module load OpenMPI/4.1.5-GCC-12.3.0 2>/dev/null
    {
        gcc --version | head -1
        mpirun --version 2>&1 | head -1
        ucx_info -v 2>/dev/null | head -2
        ompi_info 2>/dev/null | grep -E "MCA pml: ucx|MCA btl: tcp"
    } | tee "$OUT/software.txt"
    ;;
summary)
    files=$(for n in $(cat "$NODES_FILE"); do done_node "$n" && echo "$OUT/$n.txt"; done)
    nf=$(echo "$files" | grep -c .)
    echo "== nodes with data: $nf of $(wc -l < "$NODES_FILE")"

    echo; echo "== every line with the number of nodes that have it ($nf = same on all the nodes)"
    cat $files | grep -vE "^###|^DIMM: |^available:|size:" | sort | uniq -c | sort -rn

    echo; echo "== per node: RAM | DIMMs | NICs | kernel"
    for f in $files; do
        printf "%-14s | %-10s | %-3s | %-45s | %s\n" "$(basename "$f" .txt)" \
            "$(grep '^MemTotal' "$f" | cut -d' ' -f2-)" "$(grep -c '^DIMM: ' "$f")" \
            "$(grep '^NIC' "$f" | awk '{print $2, $3}' | xargs)" "$(grep '^Kernel' "$f" | cut -d' ' -f2)"
    done | sort -t'|' -k2,3

    echo; echo "== DIMMs per socket, per group (labels PROC_<socket>_DIMM_<slot> or CPU_SrcID#<socket>_...)"
    for g in $(cut -d- -f2 "$NODES_FILE" | sort -u); do
        echo "-- $g"
        for f in $(echo "$files" | grep -- "-$g-"); do
            echo "$(grep -c '^DIMM: ' "$f") DIMMs: $(grep '^DIMM: ' "$f" | awk '{print $2}' | sed -E 's/^(PROC_|CPU_SrcID#)([0-9]+).*/\2/' | sort | uniq -c | awk '{printf "socket%s=%s ", $2, $1}')"
        done | sort | uniq -c
    done
    ;;
*)
    echo "usage: $0 submit | status | software | summary"
    exit 1
    ;;
esac
