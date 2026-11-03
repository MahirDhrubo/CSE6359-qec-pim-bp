---
marp: true
paginate: true
title: DecodeCache + PIM-BP
---

<!-- OWNER: P3. Figures come from `make results` (results/*.png). Numbers marked [QUICK] are
from QUICK_GRID (surface d=5,7; p=0.001,0.003; 20k shots) and must be refreshed after the
FULL_GRID run. Every hardware number is MODELED unless the slide says otherwise. -->

# DecodeCache + PIM-BP

Keeping a real-time QEC decoder ahead of the syndrome stream

CSE 6359 · P1 · P2 · P3

---

## The problem: decoders must keep up

- Every syndrome round produces a decode window; the decoder must finish each window before the backlog grows
- What matters is the **tail**: one slow decode delays every window queued behind it
- Two decoders with the **same mean** latency can behave completely differently

![w:900](../results/backlog_explosion_synthetic.png)

<!-- Synthetic lognormal service times, same mean (0.9 of the round period).
Light tail peaks at 5 queued windows; heavy tail at 371, p99.9 latency ~90x higher. -->

---

## Our design

1. **DecodeCache**: split the syndrome into clusters; a CAM answers isolated clusters seen in training
2. **PIM-BP**: everything else runs quantized normalized min-sum BP, modeled inside SRAM
3. **OSD on CPU**: fallback when BP does not converge
4. **Backlog-aware controller**: caps BP iterations as the queue grows

<!-- TODO(P2): architecture diagram: syndrome -> splitter -> CAM -> PIM-BP -> OSD -->

---

## Method

- **Functional**: Stim circuit-level noise, rotated surface code (+ gross code [[144,12,12]]), SEED_TEST shots; cache built on SEED_TRAIN only
- **Cost model**: per-operation latency/energy, every number **MODELED** with a cited source (`op_costs.yaml`)
- **Queue simulator**: one window per round, FIFO decoder; p50 / p99 / p99.9, backlog, max sustainable rate
- Baselines: ldpc BP+OSD (**measured** CPU), conventional CPU BP (modeled), near-memory DPU and Relay FPGA (**published** numbers)

---

## Result 1: quantized BP keeps the logical error rate

![w:950](../results/ler.png)

<!-- [QUICK] Quantized vs float BP+OSD agree within 95% Wilson intervals at every point.
Triangles = 0 observed failures, drawn at the 3/N upper bound. Cache series counts hits only. -->

---

## Result 2: how often the cache answers

![w:950](../results/hit_rate.png)

<!-- [QUICK] d=5 p=0.001: 84.6% of shots, but 73.5% excluding empty syndromes (always hits).
Hit rate falls quickly with p and d (d=7 p=0.003: 5%). Say the second number, not the first. -->

---

## Result 3: BP iterations are bimodal

![w:900](../results/iters_hist.png)

<!-- [QUICK] Converging shots finish within ~20-35 iterations; 13-73% never converge and run to
the cap, then OSD. That gap is what the controller exploits. -->

---

## Result 4: latency tails per system (MODELED)

![w:900](../results/latency_cdf_surface_d5_p0.001_s2000.png)

<!-- [QUICK] The tail is a plateau: shots that need OSD (100 us placeholder) set p99 and p99.9.
42% of shots cost 0 ns in the model (empty syndrome). -->

---

## Result 5: backlog over time (MODELED)

![w:950](../results/backlog_over_time_surface_d5_p0.001_s2000.png)

<!-- [QUICK] Round period where pim_cache runs at 90% utilization (~19.6 us).
Conventional and PIM-without-cache fall behind without bound; pim_cache peaks at 337 windows;
full (cache + controller) at 39, p99.9 645 us vs 6.4 ms. -->

---

## Result 6: what each component buys (MODELED)

![w:950](../results/ablation.png)

<!-- [QUICK] Cache: 54-110x lower p99.9 than pim_nocache at three workloads, only 1.3x at d=7
p=0.003 (hit rate 5%). Controller (default thresholds 2-10): another 1.8-10x on p99.9 and
1.5-2x max stable rate. With the controller off, full == pim_cache, so the gain is the
controller's. Most aggressive threshold wins: see Limitations (LER effect of capping). -->

---

## Result 7: max sustainable round rate (MODELED)

![w:950](../results/max_rate.png)

<!-- [QUICK] With the 10 us p99.9 budget no system qualifies on current numbers (OSD placeholder
100 us, PIM BP 3.2 us/iteration). TODO: decide budget / replace with stable-rate figure. -->

---

## Data movement and energy (MODELED)

![w:900](../results/bytes_energy.png)

<!-- [QUICK] PIM moves ~2,000-11,000x fewer bytes than conventional CPU BP; energy only 1.5-3x
lower because the OSD placeholder (1 mJ/call) dominates. -->

---

## Limitations

- All hardware latency and energy is **modeled**; only ldpc BP+OSD CPU time is measured
- OSD cost is a placeholder (100 us); it sets every tail and the controller's behavior
- The controller's latency gain comes from capping BP early; the effect of capping on **logical error rate** is not modeled
- Empty syndromes cost 0 ns in the model; cache hit rate counts them as hits
- One shot = one decode window (no sliding-window decoding)

---

## Summary

- TODO after FULL_GRID: 3 numbers the audience should remember

<!-- Candidates: LER parity of quantized BP; cache hit rate excluding empty syndromes;
controller's p99.9 reduction at fixed round rate. -->
