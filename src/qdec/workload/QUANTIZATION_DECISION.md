# Day 2 quantization decision — 2026-11-02

Select **6-bit messages, range x1.5**, using TRAIN only. Keep alpha=.625,
max_iters=50, OSD-CS order 7 and the same custom OSD for float and quantized BP.
The range multiplier applies to the largest channel LLR for each workload.

The candidate domain is widths 3–8 and ranges x1/x1.5/x2/x3, with one fixed
width/range choice across QUICK_GRID. A candidate passes only when every workload
has overlapping 95% Wilson LER intervals, one-sided paired McNemar p >= .05,
and quantized OSD frequency <= 1.15 times float. A failure on any workload
rejects that candidate; missing measurements never count as passes.

## Selected candidate: 20,000 TRAIN shots per workload

| Workload | Quantized failures | Float failures | Paired p | Quantized OSD | Float OSD | OSD ratio |
|---|---:|---:|---:|---:|---:|---:|
| d5, p=.001 | 3 | 3 | 1 | .12470 | .11160 | 1.1174 |
| d5, p=.003 | 53 | 47 | .0546875 | .42265 | .37700 | 1.1211 |
| d7, p=.001 | 1 | 1 | 1 | .26315 | .24290 | 1.0834 |
| d7, p=.003 | 19 | 18 | .5 | .72500 | .68015 | 1.0659 |

All selected rows pass the rule. The d5/p=.003 paired result is close to the
threshold; this is not evidence of exact equality or a noninferiority proof.
TEST validation must be reported separately, without using it to tune settings.

## Why this width

Every 3-, 4- and 5-bit range candidate has at least one failing TRAIN workload.
At 6 bits, x1, x2 and x3 fail somewhere; x1.5 passes all four points. Therefore
6 bits is the smallest passing width **within the declared domain**. There is
no need to test 7 bits to establish this bounded minimum. This does not claim
optimality over 2-bit quantization, other ranges, different alpha, or other codes.
The old 8-bit x1 setting remains a preserved comparison, not a setting chosen
from the new TEST results.

## Evidence and reproduction

The d5 rows come from the existing `data/traces/quant_sweep_*_s1000.csv` files.
The new d7 rows are in `data/traces/day2_train_6b/`, with input/output hashes and
settings in adjacent JSON files. Both new input/output hash sets were verified
on recovery. Earlier d5 CSVs lack full original-run provenance; retain that
limitation when presenting the selection.

```sh
.venv/bin/python scripts/p1_audit_quantization.py --additional-dir data/traces/day2_train_6b
```

To independently repeat the selected candidate, run the following command for
each `(d,p)` in `(5,.001)`, `(5,.003)`, `(7,.001)`, `(7,.003)`, using a fresh output
directory. The script refuses to overwrite a saved experiment.

```sh
.venv/bin/python scripts/p1_quant_sweep.py --d 7 --p .003 --seed train --bits 6 --ranges 1.5 --output-dir data/traces/repeat_selected_train
```

Saved TEST results were inspected in earlier iterations. This project is not
a fully blinded study. The new 6-bit configuration was frozen from TRAIN before
its TEST traces were generated. The pending TEST comparison is a validation of
that choice, not another configuration search.
