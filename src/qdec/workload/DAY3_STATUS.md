# P1 Day 3 execution plan and checkpoint — 2026-11-03

## Starting state

The clean P1 branch started at `d9ec9b8`, also the verified current remote main.
All three Day 2 PRs are merged. P2 now has real cache and cost-model paths; P3's
queue/controller/results tooling is merged. P1 already uses real shots/graphs,
so there was no mock upstream dependency to replace. No P2/P3 package is imported.
The Day 2 6-bit x1.5 settings remain frozen; new distances and gross are measured,
not assumed to inherit the QUICK_GRID LER result.

## Tasks and status

1. **Start full inputs and decoding:** done. A separate process generated all
   23 graph sets and 46 TRAIN/TEST shot files in `data/day3_full/data/`. The
   original full profile has 2.15 million TEST shots. Data-only verification
   passed for every tag; see `data/day3_full/verify_inputs.log`. A full-size
   functional run started and completed d5/p=.001 before the runtime fallback.
2. **Gross circuit:** implemented and tested in `6b8ba97`. See `gross.py` and
   `tests/p1_test_gross.py`. The 288-qubit Z-memory circuit has 144 data qubits,
   12 logical Z observables, repeated ancilla syndrome extraction and the same
   noise convention as this project's surface circuit. The seven-layer schedule
   and code construction follow the authors' reference, but the idle-noise and
   memory-boundary conventions are explicitly our own. Rank/commutation,
   logical independence, noiseless detection, seed and graph tests pass.
3. **Deadline fallback:** active, as authorized by the task. Single-shot probes
   took about .94 seconds for d13/p=.005 and 1.75 seconds for gross/p=.003 in
   quantized decoding alone. These are rough throughput samples, not hardware
   cost numbers or reliable runtime forecasts. The full nominal shot counts
   imply days of computation; they are not close to a midday completion.
   Preserve all distances and p values, reduce each workload to 1,000 shots in
   a separate directory, and label the result **REDUCED PROFILE**. No shared
   grid constants change. The full raw data and completed trace remain intact.
4. **Reduced sweep:** running in `data/day3_reduced_1000/`. `generation.log` and
   `functional.log` record actual progress. The process uses .venv, real data,
   one BLAS thread, and per-workload provenance checks for resumption. Completed
   checkpoints are not redone. A current workload is not checkpointed per shot.
5. **Statistics and verification:** P1 scripts now provide LER-vs-p numbers,
   Wilson CIs, paired comparisons and iteration-count histograms. Missing traces
   must be labeled partial. Verification checks schemas, shot counts for both
   seeds, graph dimensions/statistics, hashes and BP/OSD bookkeeping. A reduced
   run cannot pass verification as the original full-count grid.
6. **Measured CPU baseline, final verification, upload:** still pending. Run CPU
   measurements after functional jobs finish, on an otherwise idle machine.
   Require latency artifacts for the final P2 handoff. Upload only the verified
   coherent profile, with its count/cut manifest and limitations, not a mixture
   of full-size inputs and reduced traces.

## Commands

From the repository root, inspect progress:

```sh
tail -20 data/day3_reduced_1000/functional.log
```

From `data/day3_reduced_1000/`, only when no earlier functional process is active:

```sh
../../.venv/bin/python ../../scripts/p1_run_functional.py --grid full --shots 1000 --resume
../../.venv/bin/python ../../scripts/p1_check_baselines.py --grid full --all --shots 1000
../../.venv/bin/python ../../scripts/p1_verify_data.py --grid full --shots 1000 --require-latency
../../.venv/bin/python ../../scripts/p1_summarize_sweep.py --grid full --shots 1000
```

For an explicitly partial progress report, append `--allow-partial` to the last
command. Runtime reports and staged artifacts stay git-ignored. The script checks
source and input fingerprints before skipping a checkpoint; a stale checkpoint
requires investigation/a fresh output directory rather than silent reuse.

## Shared target awaiting approval

`make verify-data` does not exist on main. README.md lists Makefile as shared and
frozen; the user's instruction requires asking before editing it. The P1 verifier
is implemented independently. Proposed Makefile addition (not applied):

```make
.PHONY: verify-data
verify-data:
	$(PYTHON) scripts/p1_verify_data.py --grid $(GRID) $(VERIFY_ARGS)
```

Then the requested target can verify the staged, reduced deliverable from root:

```sh
make verify-data PYTHON=.venv/bin/python GRID=full VERIFY_ARGS="--work-dir data/day3_reduced_1000 --shots 1000 --require-latency"
```

## Handoff and interpretation

Drive destination supplied by the user:
https://drive.google.com/drive/folders/1natTRWvQUrBcMxlXdKgtoOaG_kWA6gcg
Google Drive was discoverable but not connected; no upload or team post occurred.

At 1,000 shots, zero failures still have a substantial confidence upper bound;
p99.9 is effectively an extreme order statistic, not a well-resolved tail.
Report actual counts and uncertainty, and do not claim full-count results.
Gross LER is block failure over 12 logical Z observables, versus one for surface.
Gross coordinates label a torus; P2 must not assume surface-code planar cache
isolation is validated for it. P2's guide explicitly allows a surface-only cache.

Reference construction/schedule:
https://github.com/sbravyi/BivariateBicycleCodes/blob/main/decoder_setup.py
Paper: https://arxiv.org/abs/2308.07915
