# P1 Day 2 continuation — 2026-11-01

## Preserved starting point

Day 1 is merged into origin/main (819cdd8). The four rebased Day 2 patches through
804b433 are identical to their pre-rebase versions. Retain MinSumBP, custom OSD,
the sweep tooling and functional pipeline. Shared schemas stay unchanged.
All QUICK_GRID shot/graph files, four 20,000-shot 8-bit TEST functional traces,
and four measured ldpc CPU latency datasets exist locally. Data is git-ignored;
these artifacts are not delivered by merging code.

## Iteration 1: audit existing TRAIN evidence

Run `.venv/bin/python scripts/p1_audit_quantization.py` from the repository root.
The command reads saved CSVs and prints coverage; it never modifies artifacts.
It requires full 20,000-shot TRAIN records, unique configurations and one float
reference per workload. PASS requires overlapping 95% Wilson intervals, paired
one-sided p >= .05 and quantized OSD frequency <= 1.15 times float.
This is a screening rule, not an equivalence/noninferiority proof.

Saved evidence for a single bits/range configuration across the whole grid:

- 8-bit x1 passes all four workloads. Keep it as the provisional production setting.
- All 3-, 4-, 5-bit candidates at x1/x1.5/x2/x3 fail at least one workload.
- 6-bit x1, x2 and x3 fail at least one workload.
- 6-bit x1.5 passes both d5 workloads; both d7 TRAIN measurements are missing.
- 7-bit x1/x1.5/x2/x3 are entirely unmeasured.

These conclusions cover only the specified candidate domain (3–8 bits and the
four ranges), not all possible quantizers. Legacy CSVs lack input hashes and
decoder settings metadata, so coverage is not certified run provenance.

## Iteration 2: complete selection and TEST reporting

1. Preserve existing sweeps when adding measurements: the current sweep CLI
   overwrites the workload CSV. Add a separate run destination/provenance before
   new experiments, rather than losing existing evidence.
2. Measure 6-bit x1.5 on d7 TRAIN at p=.001 and .003 (20,000 shots each).
   If it passes both, it is the smallest passing width in the specified domain.
   If it fails, screen 7-bit ranges on TRAIN, rejecting a fixed configuration
   once any workload fails; require all four workloads before accepting one.
3. Freeze the chosen settings, then evaluate TEST once for reporting. Do not
   use TEST to choose widths/ranges. Existing TEST results have already been
   inspected; disclose this history rather than claiming a fully blinded study.
4. Produce a reproducible TEST comparison report: failures, LER, Wilson CIs,
   paired discordances/p-value, OSD frequency and iteration tails, with settings
   and provenance. Existing 8-bit functional traces remain reusable if unchanged.
   Their schema contains quantized iteration/OSD fields only, so float fallback
   statistics require separate evidence; do not invent them from these traces.

## Iteration 3: P2 handoff and delivery

- The float functional reference is our float BP + our custom OSD. The measured
  bposd_cpu reference is ldpc. Current regression tests do not prove universal
  equivalence of these implementations; custom OSD differs on sorting ties.
- P2's osd_ns placeholder asks for OSD-only timing. Existing bposd_latency files
  measure the complete ldpc decode; do not substitute them as OSD-only costs.
  Agree the implementation and timing definition with P2 before introducing a
  new cross-owner artifact. Shared contract changes require user approval.
- Stored messages are quantized, but variable sums/posteriors accumulate without
  saturation in float64. P2 needs a hardware accumulator-width assumption.
- Controller repricing models earlier fallback latency; it does not validate
  LER for decoding at a reduced iteration budget.
- Run make test with PYTHON=.venv/bin/python before every small commit. Share
  commands/settings/comparison results and coordinate real-data ingestion.
  P2 cache/service construction and P3 controller remain stubs in this checkout.
  No claim that teammates switched off mocks or that the branch is merged.

Gross-code construction and FULL_GRID sweeps remain Day 3 work.
