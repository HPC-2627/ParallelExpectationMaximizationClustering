# Hardware of the nodes used by the experiments

Collected with `src/script/pemc-hwinfo.sh` (one job per node, `pemc-hwprobe.sh`) on the nodes of all the runs. This is the source of the table "Hardware Infrastructure" of the report.

## Same on all the nodes

| Item | Value |
|---|---|
| CPU | 4 × Intel Xeon Gold 6252N (Cascade Lake), 2.3 GHz (turbo 3.6 GHz, Intel catalog), AVX-512 |
| Cores per node | 4 sockets × 24 cores = 96 physical cores |
| Hyper-threading | disabled (1 thread per core) |
| NUMA | 4 NUMA nodes, one per socket (24 cores each, no sub-NUMA clustering) |
| Caches | L1d 32 KB + L1i 32 KB per core, L2 1 MB per core, L3 35.75 MB per socket |
| Memory | DDR4, 32 or 64 GB DIMMs, 6 channels per socket |
| Network | Ethernet 10 Gb/s, MTU 9000 |
| Operating system | Rocky Linux 9.8, kernel 5.14.0 (`el9`) |
| Compiler, MPI | GCC 12.3.0, Open MPI 4.1.5 (PML `ucx`, BTL `tcp`), UCX 1.14.1 |
| Not readable without root | real DIMM speed (`dmidecode`), network switch |

## Differences between the nodes

| Item | Value |
|---|---|
| Nodes used | 40: 1 in group `c10`, 18 in `c11`, 21 in `c12` |
| RAM per node | from 256 GB to 2 TB |
| NIC | QLogic FastLinQ QL41000 (`c10`, `c11`), Broadcom BCM57414 NetXtreme-E (`c12`), both at 10 Gb/s |

## Memory configurations

The number of populated channels fixes the peak memory bandwidth of a socket (at DDR4-2933, 23.5 GB/s per channel: 6 channels = 141 GB/s, 4 = 94 GB/s, 2 = 47 GB/s). For `c10`/`c11` the channels are inferred from the slot numbers (two slots per channel), for `c12` they are reported by the kernel.

| DIMMs per socket | Channels used (of 6) | DIMMs per channel |
|---|---|---|
| 12 | 6 | 2 |
| 9 | 6 | 2 on 3 channels, 1 on 3 |
| 6 | 6 | 1 |
| 8 | 4 | 2 |
| 4 | 4 | 1 |
| 2 | 2 | 1 |

With 2 DIMMs per channel Cascade Lake usually lowers the memory speed (typically from 2933 to 2666 MT/s); the real speed cannot be checked without root.
