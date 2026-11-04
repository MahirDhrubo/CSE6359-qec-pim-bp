# DecodeCache + PIM-BP: full project brief

**Purpose.** Load this into a Claude session so it understands the project well enough to teach it
and help present it. Snapshot taken 2026-11-04 from the proposal, all three team branches and P1's
data. The request behind this brief is in the last section.

**Rules for using this brief**

1. The final results are assumed to exist in `results/` (section 7). If a final number disagrees
   with a number here, the final number wins. Say which one you used.
2. Never invent a result. If a figure or table is not available, ask for it.
3. Keep every number labeled: **MEASURED** (timed on our CPU), **MODELED** (our cost model),
   **PUBLISHED** (from another paper), **SYNTHETIC** (made-up demo data), or **REDUCED** (1,000-shot
   profile) / **QUICK** (20,000-shot, small grid).
4. Section 8 lists places where the original story needs correcting. Read it before writing any
   talk content.

---

## 1. The project in one paragraph

A quantum computer produces a stream of error clues about once per microsecond, and an ordinary
computer (the **decoder**) must interpret them just as fast. If it falls behind, a queue builds and
the quantum program stalls. Averages hide this risk, because a few very slow decodes can cause the
queue to grow. We designed and evaluated a decoder with two ideas: **skip work** by answering common
error patterns from a cache, and **make the remaining work cheaper** by running the decoding
algorithm (belief propagation, BP) inside memory. A controller watches the queue and reduces effort
when it grows. We judge the system by whether the queue stays bounded, not by average speed, and we
require that accuracy does not get worse. No chip was built: correctness is simulated, hardware is
modeled, and only CPU software timing is measured.

---

## 2. Background from zero, for a non-expert audience

Each step exists because of the one before it.

| Step | Plain explanation | Analogy |
|---|---|---|
| Qubits are error-prone | Today's good hardware fails roughly once per 1,000 operations (p ≈ 10⁻³). Useful algorithms need billions or more operations, so errors are certain without protection. | Doing a very long calculation with a calculator that is wrong 1 time in 1,000 |
| We can't simply copy or inspect | Quantum data cannot be copied, and measuring it destroys it. Classical "keep three copies and vote" does not directly work. | A sealed envelope: opening it ruins the contents |
| Parity checks reveal errors, not data | Helper qubits ask "do these qubits agree?" without revealing the values. The answers form a **syndrome**, and each changed answer is a **detection event** (or "defect"). | Smoke alarms: they say where something is wrong, not what is in the room |
| Error-correcting codes spread information | One protected **logical qubit** is spread across many physical qubits. A bigger **code distance d** tolerates more errors. | Writing a message so that it survives even if some letters get smudged |
| A decoder interprets the alarms | Many different errors can trigger the same alarms, so the decoder must infer the most likely cause. This runs on an ordinary computer. | A detective working out the likely cause from a pattern of clues |
| The decoder has a deadline | Superconducting hardware produces a round of checks about every 1 µs. Most corrections can be tracked in software, but some gates (non-Clifford, such as T gates) need to know the errors first before proceeding. So the quantum program waits for the decoder. | A driver who cannot take the next turn until the navigation answers |
| Falling behind compounds | More waiting means more errors and more clues to decode. This is the **backlog problem** (Terhal). | A checkout line that grows faster than it can be served |
| Averages mislead | Two decoders with the same average can behave very differently, because one slow job delays everyone queued behind it. So we report **tail latency**: p99 is slower than 99% of jobs, and p99.9 is slower than 99.9% of them. | Two cashiers with the same average, but one sometimes takes ten minutes for one customer |

**The accuracy metric.** The **logical error rate (LER)** is how often the protected answer ends up
wrong after decoding. **Physical error rate (p)** is the hardware's raw failure rate. The aim is for
LER to be much lower than p. A faster decoder is worthless if its LER is worse.

**Why two codes?**
- **Rotated surface code:** the well-studied standard, with local checks. It needs many physical qubits.
- **Gross code [[144,12,12]]:** a newer qLDPC code from IBM. It stores 12 logical qubits in 144 data
  qubits (288 with helpers), far fewer than a surface code of comparable protection (Bravyi et al.).
  The catch: errors can trigger more than two alarms, so the fast standard decoder (matching) does
  not apply. This is why BP matters.

---

## 3. The algorithms, simply

- **Matching (PyMatching):** pairs up alarms on a graph. Fast and accurate on the surface code, but
  it does not apply to the gross code's error structure.
- **Belief propagation (BP):** alarms and possible errors exchange messages on a graph, repeatedly
  updating their beliefs ("how likely is this error?") until the beliefs explain all the alarms.
  Like neighbors gossiping until the village agrees on what happened. Each **iteration** touches
  every edge of the graph. **Min-sum** is a simplified BP that uses only compares and adds, which
  suits hardware. **Normalized** means messages are scaled down (we use α = 0.625).
- **Non-convergence:** BP can oscillate and never agree. We stop at 50 iterations.
- **OSD (ordered-statistics decoding):** the slow, careful fallback. It uses BP's final beliefs and
  solves linear algebra to force a valid answer. We use OSD-CS, order 7.
- **Quantization:** storing messages in few bits. Fewer bits means cheaper hardware, but rounding
  could harm accuracy. This is the question P1 had to answer.

---

## 4. Our proposed system

```
Syndromes -> Window buffer -> Splitter -> CAM cache --hit--> Commit (correction)
 (1 round/µs)   (queue)      (clusters)      | miss             ^        ^
                   :                         v                  |        |
              queue depth             PIM-BP in SRAM --converged+        |
                   :                         | not converged             |
              Controller ..iteration budget..:                           |
                                             v                           |
                                        OSD on CPU ----------------------+
```

1. **Splitter:** groups alarms into connected clusters.
2. **DecodeCache (contribution 1):** a content-addressable memory (CAM) looks up each cluster's
   shape in about one step and returns a stored correction. A shot is a hit only if *every* cluster
   is known, isolated, and its stored correction reproduces that cluster's alarms. Think of a front
   desk that answers common questions immediately and passes the rest to a specialist. Physics
   basis: below threshold, large clusters become exponentially rarer (percolation), so most patterns
   are small and repeat.
3. **PIM-BP (contribution 2):** on a miss, BP runs inside SRAM, using bit-serial digital operations
   where the messages are stored. It is digital so the answers match the same quantized BP on a CPU.
4. **OSD on CPU:** used when BP does not converge. It is a poor fit for in-memory hardware.
5. **Backlog controller:** gives BP fewer iterations as the queue grows.

**The prior-work gap we claim:** near-memory BP places small processors beside DRAM banks, and we
compute inside SRAM arrays. Earlier predecoders (Clique, Promatch) use fixed rules, and ours is a
learned cluster cache plus a backlog-aware controller. We evaluate with queueing metrics, not just
mean latency.

---

## 5. End-to-end workflow

We cannot fabricate a chip in 4 days, so evaluation has three layers: **is it correct?** (functional
simulation), **how fast and efficient?** (cost model), and **does it keep up?** (queue simulation).
Each stage communicates only through files whose schemas are frozen in
[contracts.py](../contracts.py).

| # | Stage | Owner | What it does | Output |
|---|---|---|---|---|
| 1 | Circuits and shots | P1 | Stim simulates noisy circuits. Separate TRAIN (seed 1000) and TEST (seed 2000) shots. One shot = d rounds = one decode window (a simplification). | `shots/<tag>.npz` |
| 2 | Decoding graphs | P1 | Convert Stim's error model into the BP graph: matrices H and L, plus error probabilities. | `graphs/*.npz, *.json` |
| 3 | Quantization choice | P1 | Sweep bit widths and ranges on TRAIN; freeze the smallest passing setting. | 6-bit, range ×1.5 |
| 4 | Functional traces | P1 | Decode every TEST shot with quantized BP+OSD, float BP+OSD and matching (surface only). One row per shot. | `traces/functional_<tag>.parquet` |
| 5 | CPU baseline | P1 | Time ldpc BP+OSD per shot, one thread: the only measured latency. | `traces/bposd_latency_<tag>.npz` |
| 6 | Cache | P2 | Build from TRAIN only; evaluate on TEST. | `traces/cache_<tag>.parquet` |
| 7 | Cost model | P2 | Price each shot on each system using cited per-operation costs. Paired OSD timing. | `costs/service_<tag>.parquet`, `iter_costs.json` |
| 8 | Queue simulation | P3 | First-come-first-served queue, one window per round period. Applies the controller for `full`. | Tails, backlog, max rate |
| 9 | Figures, slides, demo | P3, P2 | `make results`, Marp deck, demos. | `results/*`, `slides/` |

Run with `make all GRID=quick|full` (data → functional → cache → costs → results).

**Experiment grid (23 workloads):** surface d ∈ {5, 7, 9, 11, 13} × p ∈ {0.1, 0.2, 0.3, 0.5}%, and
gross × p ∈ {0.1, 0.2, 0.3}%. Nominal size: 100,000 surface and 50,000 gross TEST shots per workload.

**Fairness safeguards**
- Every system is priced on the same TEST shots.
- The cache is trained on TRAIN only, and code rejects a TEST tag or matching shots during build.
- Quantization was frozen from TRAIN before the new TEST traces were generated. Earlier TEST results
  had been seen, so **do not call the study blinded**.
- Tails are kept per shot, not just as averages.

**The seven systems compared**

| System | Latency source | What the comparison shows |
|---|---|---|
| `bposd_cpu` | MEASURED | Overall gain versus standard software |
| `bp_cpu_conventional` | MODELED | Value of in-memory compute |
| `nearmem_dpu` | PUBLISHED per-iteration time, our iteration counts | In-array versus beside-array |
| `relay_fpga` | PUBLISHED per-iteration time, our iteration counts | Positioning against the state of the art |
| `pim_nocache` | MODELED | Our engine alone; compare to show the cache's value |
| `pim_cache` | MODELED | Compare with `full` to show the controller's value |
| `full` | MODELED | Our complete system (gross code: no cache, so PIM-BP + controller) |

PyMatching is the surface-code accuracy reference.

---

## 6. Questions the project had to answer

1. **Does cheap quantized BP stay accurate?** (P1) This gate comes first: without it, the rest is pointless.
2. **How often can the cache answer, safely?** (P2)
3. **How much does in-memory BP save?** (P2)
4. **Does the full system keep up, including in the tail?** (P3)
5. **What does each component contribute?** (P3 ablation: `pim_nocache` → `pim_cache` → `full`)

---

## 7. Results

### Final outputs to read

**These were not in this checkout at the snapshot**: `results/` and `data/costs/` were empty locally,
and no cache traces existed. Get them from teammates or run `make results GRID=<profile>`. Expected
files: `summary.csv`, `max_rate.csv/.png`, `ablation.csv/.png`, `ler.png`, `hit_rate.png`,
`iters_hist.png`, `bytes_energy.png`, `backlog_explosion_synthetic.png`, `latency_cdf_<tag>.png`,
`backlog_over_time_<tag>.png`. The draft deck is at `origin/p3-queue-results:slides/slides.md`; its
numbers are tagged [QUICK], and its TODOs were still open.

### Q1: Quantized BP accuracy (P1; the strongest result)

**Decision:** 6-bit messages, range ×1.5, α = 0.625, 50 iterations, OSD-CS order 7. It is the smallest
width passing in the declared domain (3–8 bits; ×1, ×1.5, ×2, ×3). Passing means: overlapping 95%
Wilson intervals, one-sided paired McNemar p ≥ 0.05, and OSD use ≤ 1.15× float on every workload.
Details: [QUANTIZATION_DECISION.md](QUANTIZATION_DECISION.md).

20,000 TEST shots per workload (QUICK, [DAY2_RESULTS.md](DAY2_RESULTS.md)):

| Workload | Quantized failures | Float failures | Matching failures | Paired p |
|---|---:|---:|---:|---:|
| d5, p=0.1% | 4 | 4 | 4 | 1 |
| d5, p=0.3% | 36 | 37 | 67 | 0.709 |
| d7, p=0.1% | 0 | 0 | 1 | 1 |
| d7, p=0.3% | 12 | 12 | 28 | 0.623 |

REDUCED sweep: 23 of 23 workloads pass the LER screen, at 1,000 TEST shots each.

**Say:** "no significant degradation detected." **Do not say:** "identical" or "proven equal." With
zero failures in 1,000 shots, the true LER could still be up to about 0.38%.

### Q2: Cache (P2; QUICK, from P3's commit and slide notes)

- d5, p=0.1%: 84.6% of shots hit, but about 42% of shots have no alarms and always hit. **Excluding
  empty shots: 73.5%.** Use this second number.
- d7, p=0.3%: about 5%. Hit rate falls quickly with p and d.
- Gross code: no cache. Translation symmetry was not validated, so this was the planned fallback.

### Q3: In-memory BP per iteration (MODELED; I recomputed these from P2's code)

| Graph | Edges | PIM µs/iter | CPU model µs/iter | Latency ratio | PIM nJ/iter | CPU model nJ/iter | Energy ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Surface d5 | 5,004 | 3.19 | 20.0 | 6× | 129 | 4,996 | 39× |
| Surface d13 | 138,284 | 3.19 | 553 | 173× | 2,010 | 137,949 | 69× |
| Gross | 391,320 | 9.58 | 1,564 | 163× | 6,266 | 1,507,557 | 241× |

Also QUICK: PIM moves about 2,000–11,000× fewer bytes per decode. Many cost inputs still have status
`VERIFY` in `op_costs.yaml` (`origin/p2-cache-cost`), meaning they were not yet checked against the
cited table.

PUBLISHED per-iteration comparison: Relay-BP on FPGA 24 ns; near-memory DPU 174.6 µs. Both carry
caveats listed in `origin/p2-cache-cost:src/qdec/costmodel/SOURCES.md`.

### Measured CPU baseline and BP's failure rate (MEASURED / REDUCED)

Median time per shot, one thread (Apple ARM):

| Workload | ldpc BP+OSD p50 | ldpc BP+OSD p99 | PyMatching p50 |
|---|---:|---:|---:|
| d5, p=0.1% | 88.7 µs | 5.4 ms | 2.8 µs |
| d7, p=0.1% | 427 µs | 47.8 ms | 3.4 µs |
| d9, p=0.1% | 1.4 ms | 248 ms | 5.1 µs |
| d11, p=0.1% | 861 ms | 897 ms | 7.4 µs |
| d13, p=0.1% | 2.49 s | 2.86 s | 10.7 µs |
| Gross, p=0.1% | 13.6 s | 17.3 s | not applicable |

Share of shots where our quantized BP failed to converge, so OSD ran:

| | p=0.1% | p=0.2% | p=0.3% | p=0.5% |
|---|---:|---:|---:|---:|
| d5 | 14.7% | 29.0% | 43.6% | 65.3% |
| d7 | 28.8% | 54.3% | 73.2% | 92.8% |
| d9 | 43.0% | 75.5% | 92.2% | 99.5% |
| d11 | 58.7% | 90.9% | 98.4% | 99.9% |
| d13 | 73.2% | 96.3% | 99.8% | 100% |
| Gross | 97.2% | 99.9% | 100% | not run |

Iteration counts are bimodal. At d5, p=0.1%, of 1,000 shots: 401 needed 0 iterations, 444 needed 1–5,
8 needed 6–10, and 147 hit the 50 cap. Almost nothing lies between.

### Q4 and Q5: Queue and controller (P3; QUICK)

- **SYNTHETIC hook:** both decoders have mean = 0.9 of the round period. The light tail peaks at 5
  queued windows and the heavy tail at 371, with p99.9 latency about 90× higher.
- **MODELED, with OSD at a 100 µs placeholder:** at the period where `pim_cache` runs at 90%
  utilization (about 19.6 µs at d5, p=0.1%), conventional and no-cache PIM fall behind without bound;
  `pim_cache` peaks at 337 windows; `full` peaks at 39, with p99.9 645 µs versus 6.4 ms.
- **Ablation:** the cache gives 54–110× lower p99.9 at three workloads but only 1.3× at d7, p=0.3%.
  The controller adds 1.8–10× on p99.9 and 1.5–2× on stable rate.
- With a 10 µs p99.9 budget, no system qualified.

**These queue numbers need refreshing.** They predate the measured OSD cost. See trap 2 in section 8.

---

## 8. Honest verdict and traps in the story

### What we solved

- A working, tested end-to-end evaluation pipeline for 23 workloads across two codes, with fair
  TRAIN/TEST separation and tail-aware queue metrics.
- Evidence that 6-bit BP messages show no significant accuracy loss. This is what makes small
  in-memory hardware plausible.
- Evidence that a cache can serve many shots at small d and low p. Before saying "safely," check the
  cache-path LER in the final results; I did not have those numbers at this snapshot.
- A modeled 6–173× latency and 39–241× energy gain per BP iteration from in-memory compute.
- A clear attribution of which component helps, and when.

### What we did not solve (unless the final results show otherwise)

- **The 1 µs target.** One modeled PIM iteration alone takes 3.2 µs on the surface code and 9.6 µs on
  the gross code. A window spans d rounds, which gives some slack, but this target is not shown met.
- **The OSD bottleneck.** See trap 2.
- **Gross-code acceleration.** BP almost never converges there on our graph, and there is no cache.
- **Silicon validation.** Everything about hardware is modeled.

**The strongest honest framing:** *we made BP far cheaper and showed that its accuracy holds, and in
doing so found that the real barrier to real-time BP is the fallback path, not BP's iterations.*

### Traps: where the original story needs correcting

I derived traps 2 and 3 from the repository's data and code. Confirm them against the final results
before stating them publicly.

1. **"One queue stays flat, the other grows without bound" is inaccurate for our hook figure.** Both
   have a mean below the period, so by P3's own stability criterion neither diverges in the long run.
   The heavy-tailed queue **balloons to hundreds of windows**. Say "same average: one peaks at 5, the
   other at 371."
2. **"OSD is a rare fallback" is false on our data, and OSD cost changes the controller story.** OSD
   runs on 13–100% of shots, and on nearly all shots at large d. P2's later paired measurements put
   OSD at about 3.3 ms (d5) and 32 ms (d7) on an Intel CPU, not the 100 µs placeholder. Then:
   - All 50 PIM iterations cost only about 160 µs. That is ~5% of one OSD call at d5, and ~0.5% at d7.
   - So capping iterations can remove at most those small percentages from a shot that needs OSD
     anyway, and it forces OSD onto any shot that would have converged later. The 1.8–10× controller
     gain was priced with OSD at 100 µs. It should be expected to shrink a lot.
   - Rough mean for `pim_nocache` at d5, p=0.1%: 7.31 × 3.19 µs + 12.9% × 3.3 ms ≈ **449 µs**, nearly
     all of it OSD. (My arithmetic, mixing P1's trace with P2's Intel OSD timing.)
3. **"BP is memory-bound" is not what our model shows for latency.** Per CPU iteration, compute versus
   memory time is 20.0 vs 0.3 µs (d5), 553 vs 9 µs (d13) and 1,564 vs 188 µs (gross). Energy is split:
   compute dominates on surface graphs (4.2 vs 0.8 µJ at d5), while memory dominates on gross (329 vs
   1,179 µJ). Two caveats: the model uses peak L2 bandwidth, which favors the CPU, and it counts 12
   scalar instructions per edge. PIM's modeled latency gain comes largely from **massive parallelism**:
   all nodes step in lockstep, so its time follows the largest node degree rather than graph size. Say
   "BP does tiny operations on a huge amount of data, so we do them in parallel where the data lives,"
   and don't claim our model proves memory is the CPU's bottleneck.
4. **The controller's accuracy impact was not tested.** Traces validate only the fixed 50-iteration
   budget. Repricing a capped shot changes its time, not its recorded correctness.
5. **"Why not just use matching?"** On the surface code, PyMatching is about 30× to over 200,000×
   faster than ldpc BP+OSD in our medians, with similar or somewhat worse LER (at d5, p=0.3%: 67 versus
   36 failures). The honest reason for BP is the gross code and similar qLDPC codes, where matching
   does not apply.
6. **Shot counts.** The nominal grid is 2.15 million TEST shots, and the data exists, but decoding it
   would take days. The sweep is a **REDUCED 1,000-shot profile**, so p99.9 is effectively the
   worst-case shot, and zero failures is weak evidence. Never present this as the full grid.
7. **Gross-code caveats.** LER is block failure over 12 logical Z observables, compared with one for
   surface, so the raw LERs are not directly comparable. It is Z-memory only, with our own noise and
   boundary conventions, so it does not reproduce IBM's paper. "12" is the code's known distance, not
   measured here.
8. **Timing comparisons cross machines.** P1 timed on Apple ARM (the sidecar records only "arm"); P2
   timed OSD on Intel. `bposd_cpu` energy assumes 10 W.
9. **Proposal items not found in code:** injected-burst testing, Sinter (we used our own Wilson and
   McNemar statistics) and NeuroSim/MNSIM (we used literature-based per-operation costs). Do not
   claim these. The Limits slide should state correlated noise was **not tested**.

---

## 9. Limits to state plainly

- All hardware latency and energy is modeled; many inputs are still marked `VERIFY`.
- PIM assumes an optimistic routing network that moves a full row per step.
- OSD's cost is a single median in the service model, which hides its own tail.
- Empty syndromes are priced at 0 ns BP.
- One shot = one window. There is no sliding-window decoding. The queue feeds one window per period,
  though each window contains d rounds, so don't compare window times directly with 1 µs without
  explaining this.
- Memory experiment only, not a full algorithm with logical gates.

---

## 10. Talk structure and my segments

Planned talk (30 minutes):

| Minutes | Segment | Speaker | Content |
|---|---|---|---|
| 0–3 | **Hook** | **P1** | Same average, very different queue |
| 3–7 | **Background** | **P1** | Qubits → syndromes → decoder; BP in one picture |
| 7–10 | Insight 1 | P2 | Most clusters are small and repeat |
| 10–13 | Insight 2 | P2 | BP's data and operations per iteration (see trap 3) |
| 13–18 | Architecture | P2 | Pipeline |
| 18–25 | Results | P3 | LER, tails, backlog, rate, hit rate, energy, ablation |
| 25–27 | Demo | P3 | Baseline versus ours, same stream |
| 27–30 | **Limits and Q&A** | **P1** | Modeled, not measured; correlated noise; qLDPC; future FPGA |

Suggested flow for the opening seven minutes: open with the checkout-line figure and no quantum terms →
"this line is a quantum computer's decoder" → why qubits need correction → alarms, not answers → the
detective and the deadline → BP as gossip, with a small graph animation → hand off with "most clues
are simple, and that gives us two ideas: skip work and make work cheaper."

**Questions to prepare for**

- *Why should anyone care if a quantum computer isn't useful yet?* Decoding is a hard requirement of
  the error-corrected machines that algorithms need, and it is a classical systems problem we can
  study now.
- *Did you build hardware?* No. Correctness is simulated, costs are modeled from cited sources, and
  only software CPU time is measured.
- *Why not use matching or Relay-BP?* See trap 5. Relay-BP is the state of the art, and we compare
  against its published iteration time, not its accuracy.
- *Can a cache hit give a wrong answer?* It can, but there are safeguards: isolation, replay against
  the alarms, and dropping entries that often failed on TRAIN. Check the cache-path LER in final results.
- *Is 6 bits proven safe?* No. No significant loss was detected on our tested settings.
- *Why is OSD so common?* BP often fails to converge on these circuit-level graphs, especially as
  d and p grow. That is the main finding to discuss, and a reason methods like Relay-BP exist.

---

## 11. Sources

**Tools:** Stim, PyMatching, `ldpc` (Roffe et al.). **Backlog and tails:** preprints.org/manuscript/202608.1425
(v1, v2); Beverland IPAM talk (ipam.ucla.edu/abstract/?tid=21717); Terhal's "backlog problem."
**BP:** Relay-BP, IBM, arXiv:2506.01779; FPGA: Maurer et al., arXiv:2510.21600; DPU near-memory: Seo et al.,
arXiv:2608.27901. **Code:** Bravyi et al., arXiv:2308.07915. **Predecoders:** Clique (arXiv:2208.08547);
Promatch (arXiv:2404.03136). **Costs:** Horowitz, ISSCC 2014; Aga et al. "Compute Caches," HPCA 2017; Eckert
et al. "Neural Cache," ISCA 2018; Pagiamtzis & Sheikholeslami, JSSC 2006. Check each citation before
putting it on a slide; several were collected quickly and are only preprints.

**Companion docs:** Systems Proposals for Real-Time QEC Decoding (claude.ai/artifact/PW6q5QR7CWch9ASUEpjdta);
DecodeCache: BP Compatibility, Physics, and STAR (claude.ai/artifact/73Av67H2kuK8xf2fBAxzMM). The proposal
is `Project Proposal DecodeCache + PIM-BP.pdf` in the repository root. Team guides are in `docs/guides/`.

---

## 12. Who I am and what I need from you

**I am P1 (Workload, quantized BP and baselines; branch `p1-workload-bp`).** I produced the ground
truth that the rest of the team's work stands on.

**What I completed**
- Stim circuits and shots for the rotated surface code, with separate TRAIN and TEST seeds.
- The [[144,12,12]] gross-code memory circuit, from the published construction and schedule, with tests
  for rank, commutation, logical independence and noiseless detection.
- Decoding graphs from Stim's error model.
- Our own quantized and float normalized min-sum BP, plus our own OSD-CS.
- Baselines: ldpc BP+OSD with per-shot measured CPU latency, and PyMatching.
- The quantization study: the sweep, the pass rule, and the TRAIN-frozen 6-bit ×1.5 choice, validated on
  four 20,000-shot TEST workloads.
- Functional traces and reports for all 23 workloads at the REDUCED 1,000-shot profile, with resumable
  runs, provenance checks, verification, confidence intervals, paired tests and iteration histograms.
- The hand-off package for P2 and P3.

**Status at this snapshot, so you don't overstate my work**
- The original full-count sweep was not finished: inputs were generated and verified, but only d5, p=0.1%
  was decoded. The REDUCED profile is the deliberate, labeled fallback.
- The measured latency for gross, p=0.3% was still running, so the package is marked PARTIAL.
- My two Day 3 commits (`6b8ba97`, `f63edf1`) were local only, and P2's Day 3–4 and P3's Day 3 work were
  on their own branches, not yet merged to `main`.
- The Day 4 fresh-clone reproducibility run had not been done.

**My job as a presenter.** I give the opening (hook and background) and the closing (limits and Q&A). My
task is to **make the motivation very clear, with background knowledge easily understandable to any
non-domain audience.** Someone with no quantum or hardware knowledge should finish my section able to say
why a decoder has a deadline, why averages are misleading, and why skipping work and making work cheaper
are the two natural ideas.

**What I need from you, Claude:** using this brief and the final results,

1. Teach me the material until I can explain it without notes, starting from section 2, and quiz me.
2. Give me all the content I need: my slide-by-slide outline, speaker script, analogies, diagrams to
   draw, and answers to likely questions.
3. Use plain words first and introduce a technical term only after the idea is clear.
4. Keep every claim honest, following the rules at the top and section 8. Flag anything I say that goes
   beyond the evidence.
