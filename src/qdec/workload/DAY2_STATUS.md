# P1 Day 2 continuation — updated 2026-11-02

## Live TEST checkpoint after f620920

The frozen production settings are 6 bits / x1.5. Fresh TEST runs write to
`data/traces/day2_test_6b/`; the canonical `data/traces/functional_*` files still
hold the preserved 8-bit runs until all four new workloads validate.

- d5/p=.001 completed: quantized/float failures 4/4, paired p=1, OSD .12905/.1168.
- d5/p=.003 completed: quantized/float failures 36/37, paired p=.70947265625,
  OSD .4275/.3825.
- d7/p=.001 and d7/p=.003 jobs launched; completion remains to be checked.

On resumption, inspect the separate directory and verify its sidecars before
rerunning anything. The commands below were launched once for each of the four
QUICK_GRID `(d,p)` pairs; rerun only missing/invalid workloads after checking
that no previous process is still active:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/p1_run_functional.py --d 7 --p .003 --output-dir data/traces/day2_test_6b
.venv/bin/python scripts/p1_report_day2.py --directory data/traces/day2_test_6b
```

The full report requires all four outputs and verifies sidecar hashes/settings.
The last full suite at this checkpoint passed 162 tests with 1 expected failure.

## Recovery checkpoint: iteration 2

The previous session completed six commits through 420e67f. The worktree was
clean on recovery; no code needed to be redone. The last full suite reported
162 passed and 1 xfailed. Sweep preservation/provenance, multi-directory audit,
functional validation/provenance, TEST reporting and the proposed OSD-only
benchmark are implemented. The timing proposal is explicitly not team-approved.

Both long-running d7 TRAIN jobs finished successfully. Input and CSV hashes in
their sidecars were verified against disk on 2026-11-02. Results:

| TRAIN workload | 6-bit x1.5 failures | Float failures | Paired p | Quantized / float OSD |
|---|---:|---:|---:|---:|
| d7, p=.001 | 1 | 1 | 1.0 | .26315 / .2429 |
| d7, p=.003 | 19 | 18 | .5 | .725 / .68015 |

Together with saved d5 TRAIN measurements, 6-bit x1.5 passes all four workloads.
It is the smallest passing width in the declared 3–8-bit, x1/x1.5/x2/x3 domain.
7-bit testing is unnecessary to establish that bounded minimum. Settings were
frozen from TRAIN on recovery, before starting new TEST generation. Existing
8-bit TEST traces remain intact until the new traces are validated.

Reproduce the selection audit:

```sh
.venv/bin/python scripts/p1_audit_quantization.py --additional-dir data/traces/day2_train_6b
```

Remaining: generate and validate 6-bit TEST traces, publish the comparison and
settings in this P1-owned directory, and prepare the real-artifact handoff.
P2 adoption, team communication and merging have not occurred.

The sections below preserve the initial audit and plan as historical context;
the recovery checkpoint above supersedes their pending-measurement statements.

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
