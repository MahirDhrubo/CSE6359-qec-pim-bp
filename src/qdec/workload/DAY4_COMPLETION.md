# P1 final checkpoint — 2026-11-04

## Complete P1 deliverables

- 23 reduced workloads, 1,000 shots for each of TRAIN and TEST: 46 shot files,
  23 graph NPZ/JSON pairs, 23 functional Parquet/JSON pairs, 23 CPU latency NPZ/JSON
  pairs. No missing P1 workload artifacts remain. The CPU process exited 0.
- Complete verification including required latency passed on all 23 tags.
- Functional report: `DAY4_LER_REPORT.md`; CPU report: `DAY4_CPU_REPORT.md`.
  All 23 per-workload LER screens pass. This does not establish equivalence.
- Presentation material: `DAY4_PRESENTATION_NOTES.md`; procedure and environment:
  `DAY4_RUNBOOK.md` and `day4_constraints.txt`.
- The partial package was delivered by the user. The final CPU update and complete
  replacement archive are prepared locally; this does not mean they were uploaded.

## Reproduction evidence actually executed

A separate local clone at c8b156f00ed7478211364c71e081c9f67f3087d3 supplied the P1
implementation. The existing Python 3.13.1 .venv supplied dependencies, with
PYTHONPATH selecting that clone's src/. This was a clean source checkout replay,
not a fresh environment install. Recorded functional source hashes match.

1. Verify all 23 frozen workloads including CPU measurements from that checkout:
   PASS. Regenerate the complete LER/histogram report: text exactly matches
   DAY4_LER_REPORT.md. The original frozen files were not modified.
2. Independently regenerate graphs and 1,000 TRAIN plus 1,000 TEST shots for surface
   d5/p=.003, surface d13/p=.005, and gross d12/p=.003: all arrays/dtypes, sparse
   graph values and graph statistics match the frozen artifacts.
3. Decode 16 evenly spaced TEST shots for each selected tag: all functional
   columns match the corresponding frozen rows (48 decoded rows total).
   IDs, source revision, versions and outcomes are in DAY4_REPRODUCTION.json.
   The executable check is scripts/p1_reproduce_subset.py.

Command from a checkout root (reference must contain data/):

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/p1_reproduce_subset.py \
  --reference data/day3_reduced_1000 --report data/subset_reproduction.json
```

This check uses scratch storage and does not overwrite the reference. It can take
several minutes, especially for gross. Do not rerun it merely to read the report.
No complete 23-workload simulation rerun or second CPU campaign was performed.
No fresh dependency installation or P2/P3 end-to-end replay is claimed.

## Handoff and integration

Original Day 3 P1 code merged via PR #8. Latest main at this checkpoint is 135b264,
which also contains P2/P3 work. Final P1 docs and reproduction script are on the
P1 branch for a follow-up merge; shared contracts/Makefile were not edited.

Existing recipients apply p1_day3_final_cpu_update.zip to the delivered partial
package. New recipients use p1_day4_complete_reduced_1000.zip. Both represent the
same 23-workload reduced dataset; do not combine with nominal full-count inputs.
Verify SHA256SUMS and run the required-latency command in DAY4_RUNBOOK.md.
The complete package manifest records the source/doc revision and file inventory.

P2 should rerun its cost build with the final gross CPU measurement, retaining
its OSD calibration provenance. P3 should regenerate affected service-based
results and integrate the P1 presentation material. Main's P2 cache is surface-only;
gross cache comparisons must not be presented as implemented. The new cost build
may measure missing OSD timings and is a separate downstream job.

CPU timings are MEASURED whole ldpc BP+OSD, not OSD-only cost. Include the provided
measurement_activity.log: pre-commit tests overlapped final gross timing. These
are host measurements under recorded activity, not an otherwise idle benchmark.
At 1,000 shots, p99.9 is near the maximum; zero failures is not zero true LER.
Gross LER is block failure over 12 logical Zs, unlike surface's single logical Z.

## Remaining team actions and acceptance boundary

- Deliver the final update or complete package; Drive is not connected to this
  session, so local preparation is not proof of upload.
- Merge the follow-up P1 branch, rerun downstream costs/results, and finalize slides.
- Agree that frozen-data replay plus selected regeneration is the time-limited
  Day 4 acceptance criterion. The guide's full fresh make-all run is not fulfilled.
- `make verify-data` remains an optional shared-file change awaiting approval.
  Direct verification already works; it is not a blocker for the P1 handoff.
