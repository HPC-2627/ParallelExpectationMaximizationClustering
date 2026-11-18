#!/bin/bash
# Prints the hardware of the node where it runs, one "key: value" line per item: the data of the
# table "Hardware Infrastructure" of the report. It is the body of the small PBS jobs submitted
# by src/script/pemc-hwinfo.sh (one per node, output in results/hwinfo/<node>.txt), but it can
# also be run by hand on any node. It needs no root: what only root can read (dmidecode, the real
# speed of the DIMMs) is reported as "not allowed".
#PBS -q shortCPUQ

echo "### $(hostname -s)"

# Operating system and kernel
echo "OS: $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "Kernel: $(uname -r)"

# CPU model, sockets, cores per socket, threads per core (1 = no hyper-threading),
# NUMA nodes (= sockets, or more with sub-NUMA clustering) and caches
lscpu | grep -E "^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)|NUMA node|L1d cache|L1i cache|L2 cache|L3 cache|CPU max MHz|CPU min MHz)"
grep -qw avx512f /proc/cpuinfo && echo "Flag: avx512f"

# RAM: total, and per NUMA node
echo "MemTotal: $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
numactl -H 2>/dev/null | grep -E "^available|size:"

# DIMMs seen by the kernel (EDAC driver, readable without root): slot label, size, type.
# Labels: PROC_<socket>_DIMM_<slot> on c10/c11 (12 slots per socket, 2 per channel),
# CPU_SrcID#<socket>_MC#<m>_Chan#<c>_DIMM#<d> on c12 (channel explicit). 6 channels per socket.
for d in /sys/devices/system/edac/mc/mc*/dimm*; do
    [ -e "$d/size" ] || continue
    echo "DIMM: $(tr ' ' _ < "$d/dimm_label") $(cat "$d/size")MB $(cat "$d/dimm_mem_type")"
done

# Real speed of the DIMMs: only dmidecode knows it, and it usually needs root
out=$(dmidecode -t memory 2>/dev/null)
if [ -n "$out" ]; then
    echo "$out" | grep -E "^\s+(Size|Type|Speed|Configured Memory Speed):" | grep -v "No Module" \
        | sed 's/^\s*//' | sort | uniq -c | sed 's/^/dmidecode: /'
else
    echo "dmidecode: not allowed"
fi

# CPU frequency: governor, driver, turbo (empty if Linux exposes no frequency driver)
echo "Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "Freq driver: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null)"
echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"

# Network: every interface that is up (name, link speed in Mb/s, MTU, model of the card),
# and InfiniBand if present
for i in /sys/class/net/*; do
    name=$(basename "$i")
    [ "$name" = lo ] && continue
    [ "$(cat "$i/operstate" 2>/dev/null)" = up ] || continue
    pci=$(basename "$(readlink -f "$i/device")")
    echo "NIC: $name speed=$(cat "$i/speed" 2>/dev/null)Mb/s mtu=$(cat "$i/mtu") model=$(lspci -s "$pci" 2>/dev/null | cut -d: -f3- | xargs)"
done
ls /sys/class/infiniband 2>/dev/null | sed 's/^/InfiniBand: /'
