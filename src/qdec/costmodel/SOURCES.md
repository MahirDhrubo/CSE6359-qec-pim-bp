# Cost-model sources

Generated from `op_costs.yaml` by `scripts/p2_cost_sources.py`; do not edit by hand
(`tests/p2_test_op_costs.py` fails when this file is stale).

Labels: **MODELED** = our model built on the cited per-operation number; **PUBLISHED** = the
paper's own reported number; **MEASURED** = timed on our CPU. Status VERIFY = value proposed
from the source and still to be checked against the cited table; CHECKED = read off the source.

## Slide table

| Operation | Latency | Energy | Unit | Source | Status |
|---|---|---|---|---|---|
| `sram_xor` | 1 ns | 40 pJ | per 256-bit row step | Aga et al., HPCA 2017; Horowitz, ISSCC 2014 | VERIFY |
| `sram_min_search` | 1 ns | 40 pJ | per bit step of an MSB-first min search (n-bit min = n steps) | Eckert et al., ISCA 2018; Horowitz, ISSCC 2014 | VERIFY |
| `bitserial_add` | 1 ns | 40 pJ | per bit step (n-bit add = n+1 steps) | Eckert et al., ISCA 2018; Horowitz, ISSCC 2014 | VERIFY |
| `convergence_or_reduce` | 2 ns | 45 pJ | per check of all syndrome bits in one subarray (row step + OR tree) | Aga et al., HPCA 2017 (our estimate) | VERIFY |
| `cam_search` | 1 ns | 0.001 pJ | latency per search; energy per entry-bit compared (1 fJ/bit/search) | Pagiamtzis & Sheikholeslami, JSSC 2006 | VERIFY |
| `dram_read_byte` | 0.039 ns | 244 pJ | per byte streamed | Horowitz, ISSCC 2014; JEDEC JESD79-4 | CHECKED |
| `dram_write_byte` | 0.039 ns | 244 pJ | per byte streamed | Horowitz, ISSCC 2014; JEDEC JESD79-4 | CHECKED |
| `l2_byte` | 0.0052 ns | 12.5 pJ | per byte streamed to/from an on-chip L2 (read or write); used while BP fits in capacity_bytes | Horowitz, ISSCC 2014; Intel Optimization Manual | VERIFY |
| `cpu_core_ns` | 1 ns | 10000 pJ | per ns of one busy core (energy_pj per ns = 10 W) | Assumed (10 W per busy core) | VERIFY |
| `cpu_op` | 0.333 ns | 70 pJ | per scalar instruction | Horowitz, ISSCC 2014; 3 GHz assumed | VERIFY |
| `relay_fpga_iter` | 24 ns | 1.3926e+06 pJ | per BP iteration; energy = 24 ns x 58.025 W total FPGA power | Maurer et al., arXiv:2510.21600 | PUBLISHED |
| `nearmem_dpu_iter` | 174600 ns | not reported | per BP iteration (X + Z, serialized) = 0.873 ms / 5 iterations | Seo et al., arXiv:2608.27901 | PUBLISHED |
| `osd_ns` | 100000 ns | 1e+09 pJ | per OSD call on the CPU (FALLBACK only; see source) | Measured here (paired ldpc timing); constant = fallback | VERIFY |

## Measured on our CPU

- **bposd_cpu**: ldpc BP+OSD wall clock per shot, one thread (P1, `p1_check_baselines.py`, TRACES_DIR/bposd_latency_<tag>.npz/.json).
- **OSD fallback** (every BP system): median of paired per-shot ldpc timings, t(BP+OSD) - t(BP) on the same non-converged syndrome (P2, `costmodel/osd_timing.py`, COSTS_DIR/osd_<tag>.json). The `osd_ns` row below is only the fallback constant.
- CPU energy for both = time x 10 W (`cpu_core_ns`, assumed).

## Modeling assumptions

- **PIM-BP** (`pim.py`): bit-serial, transposed Neural Cache layout on 256x256 SRAM subarrays; all subarrays step in lockstep, so latency follows the max check/variable degree. Accumulators are wider than the 6-bit messages. Moving messages between check and variable layouts assumes a network that routes a full 256-bit row per step (optimistic).
- **CPU baseline** (`conventional.py`): int8 messages, int32 edge lists, 12 scalar instructions per edge per iteration; bytes are priced at L2 cost while the working set fits 1 MB, else at DRAM cost; latency = max(compute, memory) (roofline).
- **Cache front end** (`service.cache_path_cost`): splitter + keys on the controller core (k(k-1) + 4k scalar instructions for k defects, paid by every shot) plus one CAM search per looked-up cluster; search energy = entries x key bits x energy per bit.
- **Published baselines** (`nearmem_dpu`, `relay_fpga`): the paper's time per BP iteration applied to our iteration counts, with the same CPU OSD fallback as every other system.
- **Gross code**: no cache (translation symmetry untested); FULL = PIM-BP + controller.

## Full citations

- **`sram_xor`**: Aga et al., "Compute Caches," HPCA 2017 (in-place bitwise ops on SRAM bitlines); energy derived from Horowitz, "Computing's Energy Problem," ISSCC 2014 (8 KB SRAM read). *Note:* Check the Compute Caches latency/energy table for the bitwise-op cost relative to a read.
- **`sram_min_search`**: Eckert et al., "Neural Cache: Bit-Serial In-Cache Acceleration of Deep Neural Networks," ISCA 2018 (bit-serial comparison, used for max-pooling); energy as for sram_xor. *Note:* Confirm Neural Cache's per-step cost for comparison/reduction ops.
- **`bitserial_add`**: Eckert et al., "Neural Cache," ISCA 2018 (bit-serial add takes n+1 cycles); energy as for sram_xor. *Note:* Confirm Neural Cache's array cycle time and per-cycle energy; 1 ns is a conservative pick.
- **`convergence_or_reduce`**: Aga et al., "Compute Caches," HPCA 2017 (wired-OR / search-style reduction on bitlines); estimate = one row step + small OR tree. *Note:* Estimate, not a table value. Replace with a direct number if you find one.
- **`cam_search`**: Pagiamtzis and Sheikholeslami, "Content-Addressable Memory (CAM) Circuits and Architectures: A Tutorial and Survey," IEEE JSSC 41(3), 2006. *Note:* Check the survey's reported energy/bit/search and search time for the design you cite.
- **`dram_read_byte`**: Energy: Horowitz, ISSCC 2014 (DRAM 1.3-2.6 nJ per 64-bit access, midpoint / 8 bytes). Latency: 1 / 25.6 GB/s, one DDR4-3200 x64 channel (JEDEC JESD79-4). *Note:* Horowitz is 45 nm-era DRAM; modern DDR4/HBM is lower energy. Say so on the slide.
- **`dram_write_byte`**: Same as dram_read_byte (writes modeled at read cost). *Note:* Symmetric read/write is a simplification.
- **`l2_byte`**: Energy: Horowitz, ISSCC 2014 (1 MB cache, 64-bit access ~100 pJ -> 12.5 pJ/byte). Latency: 1 / (64 B/cycle at 3 GHz), peak L2 bandwidth per core in the Intel 64 and IA-32 Architectures Optimization Reference Manual (Skylake server cache table). *Note:* Peak, not sustained, bandwidth (favours the CPU baseline). Capacity 1 MB = Horowitz's array size.
- **`cpu_core_ns`**: ASSUMED single-core active power of 10 W, the same assumption as osd_ns; used only to turn MEASURED bposd_cpu wall clock into energy. *Note:* Replace with the measuring CPU's package power / active cores if available.
- **`cpu_op`**: Latency: 1 cycle at an assumed 3 GHz. Energy: Horowitz, ISSCC 2014 (~70 pJ per instruction including fetch/decode/register-file overhead; the 32-bit add itself is 0.1 pJ). *Note:* Confirm the ~70 pJ instruction-overhead figure on the Horowitz slide.
- **`relay_fpga_iter`**: Maurer et al., "Real-time decoding of the gross code memory with FPGAs," arXiv:2510.21600 (2025): Relay-BP on an AMD VU19P, 2 cycles of a 12 ns decoder clock per flooding iteration = 24 ns; 58.025 W total power after place and route (Fig. 5). *Note:* Gross-code X decoder. Reused unchanged for the surface code (assumes a fully parallel design of that size fits). Relay-BP's own iteration counts (about 20 on average at p = 1e-3 per the paper) are not used; ours are.
- **`nearmem_dpu_iter`**: Seo, Han, Hour, Kim, "High-Throughput Normalized Min-Sum Belief Propagation Decoding for Quantum LDPC Codes with Near-Memory Processing," arXiv:2608.27901 (2026): maximum sampled serialized X+Z DPU compute latency about 0.873 ms at five iterations (uPIMulator, UPMEM DPUs, 11 tasklets per DPU). *Note:* Simulated, not measured on hardware. Data-qubit noise graph of [[144,12,12]] (much smaller than our circuit-level graphs), so applying it unscaled is optimistic for the DPU. A maximum, not a mean. Energy not reported.
- **`osd_ns`**: Per workload the service tables use a MEASURED-derived value instead: the median of paired per-shot ldpc timings, t(BpOsdDecoder) - t(BpDecoder) on the same non-converged TEST syndrome, with P1's bposd_cpu settings (scripts/p2_measure_osd.py -> data/costs/osd_<tag>.json). This constant is an ASSUMED fallback used only when no measurement exists for a tag. Energy = time * 10 W (cpu_core_ns). *Note:* Measured on Intel x86 (2026-11-03), one thread: about 3.3 ms at d5 and 32 ms at d7, so this fallback is far too low for those graphs. The median ignores OSD timing tails; report the osd_<tag>.json p99 alongside.
