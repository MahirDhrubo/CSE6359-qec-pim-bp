# P1 Day 4: verify, hand off, and freeze

Status checkpoint: 2026-11-04. The user has delivered the partial package.
P1's functional and CPU baseline datasets are complete for the reduced profile.
All 23 workloads passed verification with required CPU latency. P1 data replay
passed from a separate clean source checkout using the existing .venv; this is
not a fresh-environment installation or a full fresh-simulation claim.
See DAY4_COMPLETION.md for the bounded regeneration evidence and remaining team steps.

## Frozen profile and evidence

- 23 workloads: surface d=5,7,9,11,13 at p=.001,.002,.003,.005;
  gross d=12 at p=.001,.002,.003. Rounds equal distance.
- 1,000 shots per workload per seed: 23,000 TEST and 23,000 TRAIN shots.
- TRAIN seed 1000; TEST seed 2000. Never train on TEST.
- Quantized min-sum: 6 bits, range multiplier 1.5, alpha .625, 50 iterations;
  OSD-CS order 7. The float comparison uses the same custom OSD implementation.
- Functional source revision: `ddfd0a2` (P1 branch after Day 3 rebase).
  All per-file source hashes in the 23 sidecars were checked against this code.
  Documentation-only commits do not change these hashes.
- Actual generating environment: Python **3.13.1**, despite the guide's 3.11
  convention. See `day4_constraints.txt` for observed package versions; this is
  an environment snapshot, not a portable lockfile. Do not silently claim 3.11
  reproduction. Capture any environment deviations and investigate mismatches.
- Delivered base: `p1_day3_reduced_1000_partial_20261104_015922.zip`.
  Includes 46 shot NPZs, 23 graph NPZ/JSON pairs, 23 functional Parquet/JSON pairs,
  and 22 CPU latency NPZ/JSON pairs. Its manifest and SHA256SUMS identify the files.
- Final update pair (now available): `data/traces/bposd_latency_gross_d12_p0.003_s2000.npz` and `.json`.
- Raw data is git-ignored; the committed report is `DAY4_LER_REPORT.md`.

## Fast path from a clean checkout

Use a separate checkout so Day 2, nominal full-count and reduced datasets cannot
silently overwrite one another. Record the exact published P1 Day 4 commit and
integrated P2/P3 commit used; the branch name alone is not a frozen revision.

The separate-checkout replay used the existing .venv. For a genuinely fresh
environment (not tested here), create Python 3.13.1, then from the checkout root:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -c src/qdec/workload/day4_constraints.txt -e '.[dev]'
.venv/bin/python -V
.venv/bin/python -m pip freeze
```

Extract the delivered ZIP into `handoff/` (which must contain `data/`,
`manifest.json` and `SHA256SUMS.txt`). Keep the archive unchanged as evidence.
Check its original checksums before applying any later update:

```sh
(cd handoff && shasum -a 256 -c SHA256SUMS.txt)
.venv/bin/python scripts/p1_verify_data.py --grid full --shots 1000 --work-dir handoff
(cd handoff && ../.venv/bin/python ../scripts/p1_summarize_sweep.py --grid full --shots 1000 > regenerated_ler.md)
```

Acceptance: 23/23 functional workloads pass; table and histogram values match
`DAY4_LER_REPORT.md`. Each histogram sums to 1,000. Verify source fingerprints
and package versions against the functional JSON sidecars when checking a new
checkout; the structural verifier does not itself enforce source/version equality.
A passing LER screen is not proof of equivalence. Missing CPU files are allowed
by this first command, but malformed CPU files already present must fail.

After the final latency pair has been delivered and its checksums verified:

```sh
.venv/bin/python scripts/p1_verify_data.py --grid full --shots 1000 --require-latency --work-dir handoff
```

This must succeed for all 23 workloads before declaring a complete P1 handoff.
Do not run `make all GRID=full`: it regenerates nominal full-count inputs, does
not invoke CPU measurements, and does not reproduce this reduced profile.
`make verify-data` is still absent; direct P1 commands require no shared edit.

## P2 and P3 integration

P2 reads this same `handoff/data/` to build cache and cost artifacts. From `handoff/`
with the integrated team code installed, the existing entry points are:

```sh
../.venv/bin/python ../scripts/p2_build_cache.py --grid full
../.venv/bin/python ../scripts/p2_build_costs.py --grid full
../.venv/bin/python ../scripts/p3_make_results.py --grid full
```

These are downstream instructions, not a claim that the integrated full run has
passed. P2/P3 own missing implementations and fixes. Their real-data readers
consume actual file row counts; do not use MOCK to fill gaps. Validate every
selected (tag, system) against 1,000 TEST shot IDs and distinguish intentionally
unsupported systems from missing results. The merged P2 implementation uses a surface-only cache fallback; gross cache
rows must not be presented as an implemented cache result. The current costs
script can measure missing OSD calibration data (default 60 seconds per tag);
this downstream step can take tens of minutes and was not run by P1.
P3 must regenerate the affected service-based figures after P2 adds the final
CPU baseline rows. P1 does not edit their code or slides.

## Independent regeneration check (after timing ends)

Implemented bounded check (`scripts/p1_reproduce_subset.py`): surface d5/p=.003, surface d13/p=.005 and gross
p=.003, using the same environment and seeds. Regenerate 1,000-shot inputs for
those tags in a separate scratch directory and compare arrays (including sparse
H/L structure and values) with the delivered inputs. Keep the same sample count;
do not assume changing Stim's sample count preserves the same shot prefix.
Then decode a recorded selection of, for example, 16 saved shots per tag and
compare quantized/float failure outcomes, BP iterations/convergence and surface
matching outcomes with the corresponding frozen trace rows. Use isolated subset
files and a matching Workload.shots value. Never overwrite delivered files.

Record exact selected shot IDs, source revision, environment and outcomes.
This is a partial regeneration check, not a rerun of the 23-workload experiment.
Do not launch it concurrently with CPU timing. Full reduced-profile regeneration
is an optional multi-hour task, not a prerequisite for teammates to use the data.

## Final CPU update and freeze checklist

1. Wait for the existing baseline process to finish successfully. Do not restart
   the entire timing script; it overwrites already measured datasets.
2. Run the verifier with `--require-latency` in `data/day3_reduced_1000`.
3. Package only the missing NPZ/JSON pair, an update manifest identifying the base
   package, SHA-256 checksums, the complete verification log, and timing notes.
   Label an update COMPLETE only after all 23 measurements pass validation.
4. Retain the original partial archive. Recipients apply the pair under its exact
   `data/traces/` paths, verify the update checksums and rerun complete verification.
   Keep base and update manifests separately so the original package remains auditable.
5. P2 regenerates affected CPU service rows; P3 regenerates affected results.
6. Freeze the combined artifact inventory, code revisions and reports. A changed
   seed, count, decoder or outcome requires a new version and downstream rebuild.

CPU latency is measured whole **ldpc BP+OSD** decode time, not custom Python OSD-only
cost. CPU timings are not expected to reproduce bit-for-bit. The original final
measurement overlapped pre-commit tests on Nov 4, 02:47–02:50 Chicago time; see
`data/day3_reduced_1000/measurement_activity.log` for that and later test activity.
Include this log in the final update and disclose possible timing contention.
Do not silently drop slow samples or claim an otherwise idle-machine benchmark.

## Honest Day 4 completion statement

Use DAY4_COMPLETION.md to distinguish executed checks from procedures.
The fallback statement after replay and selected checks pass is: "Results reproduced from the frozen reduced dataset, with selected
independent regeneration checks; no complete fresh simulation rerun."
This differs from the guide's full fresh-run requirement and needs team agreement.
The original P1 Day 3 implementation is merged into main (PR #8); current
main also includes P2/P3 updates through 135b264. Remaining team steps: merge
this final P1 documentation/check script, downstream replay, slide integration
and delivery of the final data update. The partial package has already been delivered by the user.
