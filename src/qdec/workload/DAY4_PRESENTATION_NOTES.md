# P1 LER material for P3

Source: `DAY4_LER_REPORT.md`, regenerated from all 23 real functional traces on
2026-11-04. Profile: REDUCED, 1,000 TEST shots per workload. This is material for
P3 to integrate into slides, not a change to P3-owned slide files.

## Accuracy slide

Suggested title: **6-bit BP: no significant LER degradation detected in this sweep**.

- Same shots and custom OSD-CS order 7 for quantized and float BP; alpha .625,
  max 50 iterations; quantization frozen at 6 bits with range multiplier 1.5.
- All 23 per-workload screens pass: overlapping 95% Wilson intervals and paired
  one-sided p >= .05. This is a screening rule, not an equivalence test or proof.
- Example: surface d5, p=.005: 9/1,000 quantized failures vs 10/1,000 float.
- Stress case: gross p=.003: 65/1,000 (6.5%) quantized vs 63/1,000 (6.3%) float;
  quantized 95% CI approximately 5.13–8.20%, float 4.95–7.98%, paired p=.44784.
- Zero failures at N=1,000 still gives a Wilson upper limit around 0.383%.
  Do not say zero LER or guaranteed accuracy.

Speaker wording: "We used the same error samples and fallback decoder to compare
6-bit messages with floating-point messages. We did not detect a significant
accuracy loss on this reduced sweep. The sample size limits what that establishes."

## Iterations and fallback slide

Suggested title: **OSD fallback dominates the difficult workloads**.

- Quantized OSD use rises from 14.7% at surface d5/p=.001 to 100% at d13/p=.005.
- Gross OSD rates are 97.2%, 99.9% and 100% at p=.001,.002,.003.
- p99 iteration count is 50 for all 23 workloads. Histograms include zero-iteration
  shots and sum to 1,000 per workload; this is not a histogram of averages.
- These are functional observations, not hardware speedups. OSD latency assumptions
  materially affect the modeled end-to-end systems result and belong in limitations.

## Labels and boundaries

Functional LER is simulated. CPU ldpc latency is MEASURED. P2 hardware costs are
MODELED; published DPU/FPGA values need their own citations and labels.
Do not attach whole ldpc BP+OSD timing to the custom OSD-only fallback price.
Gross is a 12-logical-Z block-failure metric; surface has one logical Z, so avoid
claiming a direct per-logical-qubit comparison. Matching is not applicable to gross.
Our gross circuit uses project noise/boundary conventions, not an exact reproduction
of the reference paper's experiment. Quantization screening does not validate LER
under a reduced controller iteration budget. N=1,000 cannot establish a robust
p99.9 tail; report sample count alongside any extreme percentile.

The original nominal 2.15-million-TEST-shot sweep was not completed. This report
covers every workload tag at reduced counts. Retain the independent Day 2 QUICK
validation as separate evidence; do not pool its 20,000-shot rows into this sweep.

## Handoff message (draft for the user to send)

"The P1 partial package is delivered: all 23 workloads' shots, graphs and functional
traces at 1,000 shots per seed, plus the now-complete set of 23 CPU baselines (base package plus final update). Please use this coherent
profile and preserve tag/shot_id alignment. The final gross CPU measurement is supplied as an NPZ/JSON update; rebuild
its CPU service rows and affected plots after applying it.
The LER table, histograms and presentation caveats are in P1's DAY4_LER_REPORT.md
and DAY4_PRESENTATION_NOTES.md. A fresh-clone replay procedure is in DAY4_RUNBOOK.md;
see DAY4_COMPLETION.md for executed replay and selected regeneration checks."
