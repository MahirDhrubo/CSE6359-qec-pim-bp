# DecodeCache + PIM-BP

A real-time quantum error-correction decoder study. Each decode window (one shot of syndrome
rounds) first goes to **DecodeCache**, a content-addressable memory that answers common, isolated
syndrome clusters directly. Everything else runs **quantized normalized min-sum belief propagation**,
modeled as running inside SRAM (**processing-in-memory**). OSD runs on the CPU as a fallback when BP
does not converge, and a **backlog-aware controller** caps BP iterations as the decode queue grows.
We evaluate it in three ways: a functional simulation (Stim + Python BP) for logical error rate,
a per-operation latency/energy model, and a discrete-event queue simulator for tail latency, backlog
and maximum sustainable round rate. **All hardware numbers are MODELED, not measured.** The only
measured latency is the CPU wall clock of the ldpc BP+OSD baseline.

## Setup

Requires Python ≥ 3.11. If your system Python is older, install `uv` (`pip install --user uv`)
and let it fetch 3.11.

```bash
git clone git@github.com:MahirDhrubo/CSE6359-qec-pim-bp.git && cd CSE6359-qec-pim-bp
uv venv --seed -p 3.11 .venv        # or: python3.11 -m venv .venv
source .venv/bin/activate
make install                        # pip install -e ".[dev]"
make test                           # must pass before every push
```

Pipeline (`GRID=quick|full`, `MOCK=1` uses `qdec.mocks` instead of upstream data):

```bash
make all                  # data -> functional -> cache -> costs -> results
make costs results MOCK=1 # P2/P3 outputs from mock data, no upstream needed
make lint
```

Unimplemented stages print `not implemented yet` (or `SKIPPED, missing input`) and exit 0, so
`make all` always runs to the end.

## Data flow

Every arrow is a file on disk. Nobody imports anyone else's code
(see [src/qdec/contracts.py](src/qdec/contracts.py)):

```
  P1 workload ──shots/*.npz, graphs/*.npz, graphs/*.json──► P2 cache + cost model
  P1 workload ──traces/functional_<tag>.parquet───────────► P2 cost model
  P2 cache    ──traces/cache_<tag>.parquet────────────────► P2 cost model
  P2 costs    ──costs/service_<tag>.parquet + iter_costs.json──► P3 queue sim + plots
  P1 + P2 traces ─────────────────────────────────────────► P3 LER / hit-rate plots
```

All tables go through `qdec.io.write_table` / `read_table`, which enforce the schemas.

## Branches and ownership

| Branch            | Owner | May edit                                           |
|-------------------|-------|----------------------------------------------------|
| `p1-workload-bp`  | P1    | `src/qdec/workload/`, `scripts/p1_*`, `tests/p1_*` |
| `p2-cache-cost`   | P2    | `src/qdec/cache/`, `src/qdec/costmodel/`, `scripts/p2_*`, `tests/p2_*` |
| `p3-queue-results`| P3    | `src/qdec/queue/`, `src/qdec/results/`, `scripts/p3_*`, `tests/p3_*`, `slides/` |

Shared and frozen on `main`: `src/qdec/contracts.py`, `src/qdec/mocks.py`, `src/qdec/io.py`,
`pyproject.toml`, `Makefile`, `CLAUDE.md`, `tests/test_contracts.py`.

## Guides

- [docs/WORKFLOW.md](docs/WORKFLOW.md): branching, daily routine, contract changes, milestones
- [docs/guides/P1.md](docs/guides/P1.md): workload, BP, baselines
- [docs/guides/P2.md](docs/guides/P2.md): cache and cost model
- [docs/guides/P3.md](docs/guides/P3.md): queue simulator, controller, results, slides
