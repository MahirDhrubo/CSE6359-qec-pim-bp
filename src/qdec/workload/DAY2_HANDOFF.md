# P1 Day 2 handoff

## Reproduce the inputs and functional traces

From the repository root, with the existing `.venv`:

```sh
make data PYTHON=.venv/bin/python GRID=quick
make functional PYTHON=.venv/bin/python GRID=quick
.venv/bin/python scripts/p1_report_day2.py
.venv/bin/python scripts/p1_check_baselines.py --all
make test PYTHON=.venv/bin/python
```

The first command generates both TRAIN (1000) and TEST (2000) shots. Functional
traces contain TEST only, 20,000 rows per tag; join on `(tag, shot_id)`. The report
checks complete row counts, row order, defect counts and iteration bookkeeping.
All four QUICK_GRID tags need real traces before the P2 no-mock handoff.
`data/` is ignored by Git, so teammates must regenerate or explicitly transfer
the corresponding artifacts. Do not mix older traces from a different bit width.

## Decoder meanings and hardware assumptions

- `fail_bp_osd`: our quantized normalized min-sum BP + our OSD-CS fallback.
- `fail_bp_float`: our float normalized min-sum BP + the same OSD-CS fallback.
- `fail_matching`: PyMatching on the decomposed surface-code DEM.
- `bp_iters`, `bp_converged`, `osd_ran`: quantized decoder behavior only.
- Empty syndromes converge at zero iterations. Nonconverged BP uses all 50
  iterations and invokes OSD. Convergence does not imply logical success.
- Both BP variants use alpha=.625 and OSD-CS order 7. The float comparator is
  not the independently timed ldpc decoder; sorting ties may differ in OSD.
- Edge/channel messages are quantized. Posterior sums accumulate without
  saturation in float64. P2 must model sufficient accumulator width, including
  normalization/rounding/saturation, rather than assume all arithmetic has the
  stored message width.
- Fixed 50-iteration traces validate fixed-budget LER. P3's controller repricing
  alone does not validate logical outcomes after an earlier OSD fallback.

## Measured CPU timing and the separate OSD decision

`bposd_latency_<tag>.npz` contains `shot_id` and `latency_ns` for the **complete**
ldpc `decoder.decode` call, with an adjacent JSON recording hardware, versions,
settings and count. It is suitable for the `bposd_cpu` baseline. It includes BP
and any OSD, and must not be inserted as an OSD-only per-fallback cost.

Proposed P2 decision: measure the custom OSD implementation separately on the
posteriors produced by the selected quantized BP, reporting timings conditional
on fallback, with CPU/thread settings and tails. Such measurements describe the
Python implementation; they are neither ldpc's internal OSD time nor hardware
PIM latency. P2 owns adoption of a cost assumption and its YAML source label.
No new timing column or shared interface is introduced by this handoff.

## Completion boundary

P1 can deliver tested code, real traces and the comparison report. Team Day 2
also requires P2 to consume the real traces and P3 to consume real service tables.
Those implementations are outside P1 ownership. Posting the report to teammates,
pushing/merging and verifying downstream ingestion are separate delivery actions;
this document does not assert they have occurred.
