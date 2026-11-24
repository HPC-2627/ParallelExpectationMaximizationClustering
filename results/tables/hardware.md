# Hardware of the nodes used by the experiments

Collected with `src/script/pemc-hwinfo.sh` (one job per node, `pemc-hwprobe.sh`) on the nodes of all the runs.

This is the source of the table "Hardware Infrastructure" of the report.

## Same on all the nodes

| Item | Value |
|---|---|
| CPU | 4 × Intel Xeon Gold 6252N (Cascade Lake), 2.3 GHz (turbo 3.6 GHz, Intel catalog), AVX-512 |
| Cores per node | 4 sockets × 24 cores = 96 physical cores |
| Hyper-threading | disabled (1 thread per core) |
| NUMA | 4 NUMA nodes, one per socket (24 cores each, no sub-NUMA clustering) |
| Caches | L1d 32 KB + L1i 32 KB per core, L2 1 MB per core, L3 35.75 MB per socket |
| Memory | DDR4, 32 or 64 GB DIMMs, 6 channels per socket (2, 4 or 6 populated) |
| Network | Ethernet 10 Gb/s, MTU 9000 |
| Operating system | Rocky Linux 9.8, kernel 5.14.0 |
| Compiler, MPI | GCC 12.3.0, Open MPI 4.1.5 (PML `ucx`, BTL `tcp`), UCX 1.14.1 |

## Differences between the nodes

| Item | Value |
|---|---|
| Nodes used | groups: `c10`, `c11`, `c12`|
| RAM per node | from 256 GB to 2 TB |
| NIC | QLogic FastLinQ QL41000 (`c10`, `c11`), Broadcom BCM57414 NetXtreme-E (`c12`), both at 10 Gb/s |

## Memory configurations

- **DIMM**: one RAM module (a memory stick) plugged into a slot of the motherboard.
- **Channel**: an independent path between the memory controller of a socket and its DIMMs; the channels of a socket work in parallel, so their bandwidths add up. Each socket of the Xeon 6252N has 6 channels, with 2 slots per channel (12 slots per socket).
- **Populated channel** (or "channel used"): a channel with at least one DIMM. An empty channel gives no bandwidth.
- **DIMMs per channel**: how many of the 2 slots of a channel hold a DIMM. A second DIMM on the same channel adds capacity, not bandwidth.

The table lists the configurations found on the nodes, per socket (all the sockets of a node are configured the same way): how many DIMMs are installed, how many of the 6 channels they occupy, and how many DIMMs share each channel.


| DIMMs per socket | Channels used (of 6) | DIMMs per channel |
|---|---|---|
| 12 | 6 | 2 |
| 9 | 6 | 2 on 3 channels, 1 on 3 (*) |
| 6 | 6 | 1 |
| 8 | 4 | 2 |
| 4 | 4 | 1 |
| 2 | 2 | 1 |

(*) 3 channels hold 2 DIMMs and 3 channels hold 1 (3 × 2 + 3 × 1 = 9): all 6 channels are used, so the bandwidth is full, but the channels do not have the same capacity.

