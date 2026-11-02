# P1 Day 2 status — 2026-11-02

P1 implementation and local QUICK_GRID validation are complete. Team adoption
and merging are not verified; see the remaining delivery items below.

## Recovery from the interrupted iteration

The working tree was clean. Six commits through `420e67f` had completed sweep
preservation/provenance, combined TRAIN audits, functional input validation and
metadata, TEST reporting, and the explicitly unapproved OSD timing proposal.
The two d7 TRAIN jobs also finished successfully after the last progress message.
Their input/output hashes were verified; no completed work needed to be redone.

Day 1 remains merged in origin/main at `819cdd8`. The original four Day 2 patches
survived the rebase unchanged. All new changes remain within P1-owned paths.

## Iteration outcomes

1. **Evidence audit complete:** smaller widths and missing range measurements
   were distinguished from passing evidence; saved runs are never overwritten
   by the sweep tool, and follow-up runs can be audited together.
2. **Selection and TEST validation complete:** 6-bit x1.5 was frozen from TRAIN
   in `f620920`, then all four TEST workloads were freshly decoded. It is the
   smallest passing width within the declared 3–8-bit, x1/x1.5/x2/x3 domain.
   See [QUANTIZATION_DECISION.md](QUANTIZATION_DECISION.md) for the rule and TRAIN
   evidence, including the near-threshold d5 TRAIN result and provenance limits.
3. **P1 handoff prepared:** the committed [TEST report](DAY2_RESULTS.md) contains
   counts, confidence intervals, paired comparisons, OSD ratios, iteration tails
   and artifact hashes. [DAY2_HANDOFF.md](DAY2_HANDOFF.md) gives reproduction
   commands, field meanings, hardware assumptions and the OSD timing proposal.

## Final TEST results: 20,000 shots per workload

| Workload | Quantized failures | Float failures | Paired p | Quantized / float OSD rate |
|---|---:|---:|---:|---:|
| d5, p=.001 | 4 | 4 | 1 | 1.1049 |
| d5, p=.003 | 36 | 37 | .709473 | 1.1176 |
| d7, p=.001 | 0 | 0 | 1 | 1.0842 |
| d7, p=.003 | 12 | 12 | .623047 | 1.0623 |

Every workload passes the LER screening rule and the <=1.15x OSD-frequency
criterion. This means no significant degradation detected, not proven equality
or noninferiority. Earlier TEST results had already been inspected; do not
present the project as a fully blinded study. Settings were not retuned using
these new TEST results.

The regenerated float and matching failure arrays exactly reproduce the prior
references on every shot. All new input, source and output hashes, schema/row
alignment, bit width and BP/OSD bookkeeping were checked before installation.

## Artifact locations and checks

- Canonical `data/traces/functional_<tag>.parquet` files now contain the four
  validated 6-bit TEST traces (80,000 rows total), with adjacent JSON metadata.
- `data/traces/day2_test_6b/` preserves the new validation runs.
- `data/traces/day2_previous_8b/` preserves all four original 8-bit traces.
- `data/traces/day2_train_6b/` holds the new d7 TRAIN evidence. Older d5 TRAIN
  sweeps remain in `data/traces/`; shots, graphs and ldpc baseline latencies
  were not changed.
- Last full check: `make test PYTHON=.venv/bin/python` — 162 passed, 1 xfailed.
  Scoped lint passes for all P1 source, scripts and tests.

All data directories remain git-ignored. Teammates must regenerate or explicitly
transfer artifacts; committing code does not transmit them.

## Remaining team delivery decisions

- Share the report and deliver/merge the branch. No push, merge or teammate
  message was performed in this iteration.
- P2 must confirm consumption of real functional traces; P3 must confirm real
  service-table integration. Their implementations are outside P1 ownership.
- OSD-only cost is a **proposal**, not a team-approved measurement choice.
  The user confirmed that no agreement exists. Whole ldpc BP+OSD latency must
  not be substituted for OSD-only cost. P2 owns adoption and the cost-model YAML.
- Controller repricing does not establish reduced-budget LER. Stored-message
  bit width does not define accumulator width. Both limitations are documented
  for downstream interpretation; no shared contract was changed.

Gross-code implementation and FULL_GRID sweeps remain Day 3 work.
