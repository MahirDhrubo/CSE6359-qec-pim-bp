---
marp: true
paginate: true
title: DecodeCache + PIM-BP
---

<!-- OWNER: P3. Figures come from `make results` (results/*.png). Numbers marked [QUICK] are
from QUICK_GRID (surface d=5,7; p=0.001,0.003; 20k shots each). Every hardware number is
MODELED unless the slide says otherwise. Time budget per slide in [~N min]; total ~27 min,
leaving 3 min of margin under the 30-min limit. -->

# DecodeCache + PIM-BP

Keeping a real-time QEC decoder ahead of the syndrome stream

CSE 6359 · TODO: team member names

<!-- [~0.5 min] -->

---

## The problem: decoders must keep up

- Every syndrome round produces a decode window; the decoder must finish each window before the backlog grows
- What matters is the **tail**: one slow decode delays every window queued behind it
- Two decoders with the **same mean** latency can behave completely differently

![w:900](../results/backlog_explosion_synthetic.png)

<!-- [~2.5 min] Synthetic lognormal service times, same mean (0.9 of the round period).
Light tail peaks at 5 queued windows; heavy tail at 371, and its p99.9 latency is ~90x higher. -->

---

## Our design

1. **DecodeCache**: split the syndrome into clusters; a CAM answers isolated clusters seen in training
2. **PIM-BP**: everything else runs quantized normalized min-sum BP, modeled inside SRAM
3. **OSD on CPU**: fallback when BP does not converge
4. **Backlog-aware controller**: caps BP iterations as the queue grows

<!-- [~2.5 min] TODO(P2): architecture diagram: syndrome -> splitter -> CAM -> PIM-BP -> OSD -->

---

## Method

- **Functional**: Stim circuit-level noise, rotated surface code, SEED_TEST shots; cache built on SEED_TRAIN only
- **Cost model**: per-operation latency/energy, every number **MODELED** with a cited source (`op_costs.yaml`)
- **Queue simulator**: one window per round, FIFO decoder; p50 / p99 / p99.9, backlog, max sustainable rate
- Systems: conventional CPU BP, PIM without cache, PIM + cache, full (PIM + cache + controller)

<!-- [~2 min] Results cover QUICK_GRID; FULL_GRID (d up to 13, gross code) and the measured /
published baselines (ldpc BP+OSD, near-memory DPU, Relay FPGA) were not merged in time.
Say this here rather than letting someone ask. -->

---

## Live demo

`python scripts/p3_demo.py --pause 1.5`

1. Same mean, different tails
2. Real decode-latency tails per system
3. The queue at one round rate: controller off vs on

<!-- [~3 min] Runs in ~1 s plus pauses. If anything fails: show results/demo_transcript.txt,
or play the recorded backup. Point at the sparklines: the conventional and no-cache rows climb
to full blocks; the controller row stays low. -->

---

## Result 1: quantized BP keeps the logical error rate

![w:950](../results/ler.png)

<!-- [~2 min] [QUICK] Quantized vs float BP+OSD agree within 95% Wilson intervals at every
point. Triangles = 0 observed failures, drawn at the 3/N upper bound. The cache series counts
cache hits only. -->

---

## Result 2: how often the cache answers

![w:950](../results/hit_rate.png)

<!-- [~1.5 min] [QUICK] d=5 p=0.001: 84.6% of all shots, but 73.5% excluding empty syndromes
(which are always hits). Quote the second number. Hit rate falls quickly with p and d
(d=7 p=0.003: 5%). -->

---

## Result 3: BP iterations are bimodal

![w:900](../results/iters_hist.png)

<!-- [~1.5 min] [QUICK] Converging shots finish within ~20-35 iterations; 13-73% never converge
and run to the cap, then OSD. That gap is what the controller exploits. -->

---

## Result 4: latency tails per system (MODELED)

![w:900](../results/latency_cdf_surface_d5_p0.001_s2000.png)

<!-- [~1.5 min] [QUICK] The tail is a plateau: shots that need OSD (100 us placeholder) set
p99 and p99.9 for every system. 42% of shots cost 0 ns in the model (empty syndrome). -->

---

## Result 5: backlog over time (MODELED)

![w:950](../results/backlog_over_time_surface_d5_p0.001_s2000.png)

<!-- [~2.5 min] [QUICK] Round period where pim_cache runs at 90% utilization (~19.6 us).
Conventional CPU and PIM without cache fall behind without bound; pim_cache peaks at 337
queued windows, p99.9 6.4 ms; full (cache + controller) peaks at 39, p99.9 645 us. -->

---

## Result 6: what each component buys (MODELED)

![w:950](../results/ablation.png)

<!-- [~2 min] [QUICK] Cache: 54-110x lower p99.9 than PIM without cache at three workloads,
only 1.3x at d=7 p=0.003 (hit rate 5%). Controller (default thresholds): another 1.8-10x on
p99.9, 1.5-2x max stable rate. With the controller off, full == pim_cache, so the gain is the
controller's. The most aggressive threshold wins: see Limitations. -->

---

## Result 7: max sustainable round rate (MODELED)

![w:950](../results/max_rate.png)

- Criterion: p99.9 latency (queueing + decode) ≤ 1 ms
- Every system is in the **kHz** range, far from 1 µs rounds (1 MHz)

<!-- [~1.5 min] [QUICK] Conventional CPU never meets 1 ms. PIM without cache 4-12 kHz, PIM +
cache 5-26 kHz, full 5-78 kHz (up to 3x PIM + cache). The kHz ceiling comes from the model:
3.2 us per PIM BP iteration and a 100 us OSD placeholder. Nothing meets 10 us. -->

---

## Data movement and energy (MODELED)

![w:900](../results/bytes_energy.png)

<!-- [~1 min] [QUICK] PIM moves ~2,000-11,000x fewer bytes than conventional CPU BP
(225-726 B vs 0.5-8 MB per decode); energy only 1.6-2.9x lower because the OSD placeholder
(1 mJ per call) dominates. -->

---

## Limitations

- All hardware latency and energy is **modeled**; OSD cost is a placeholder (100 µs) and it sets every tail
- The controller's gain comes from capping BP early; the effect of capping on **logical error rate** is not modeled
- Results cover QUICK_GRID (surface d=5,7) only; no gross code, no measured/published baselines
- Empty syndromes cost 0 ns in the model; the cache hit rate counts them as hits
- One shot = one decode window (no sliding-window decoding)

<!-- [~2 min] -->

---

## Summary

- **Quantized BP is free**: same logical error rate as float BP+OSD, within error bars
- **The cache answers most windows at low noise**: 73.5% of non-empty syndromes at d=5, p=0.001
- **Tails decide keeping up**: at the same round rate, the backlog controller cuts p99.9 latency up to 10x (6.4 ms → 0.65 ms) and raises the 1 ms-budget round rate up to 3x

<!-- [~1 min] Then questions. -->
