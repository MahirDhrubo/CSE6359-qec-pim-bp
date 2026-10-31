# DecodeCache + PIM-BP — rules for Claude Code

Course project, 4 days, 3 people, 3 branches. Speed matters more than polish, but
correctness of the logical-error-rate (LER) comparison matters most.

## Ownership (do not edit outside your area)
| Branch            | Owner | May edit                                           |
|-------------------|-------|----------------------------------------------------|
| `p1-workload-bp`  | P1    | `src/qdec/workload/`, `scripts/p1_*`, `tests/p1_*` |
| `p2-cache-cost`   | P2    | `src/qdec/cache/`, `src/qdec/costmodel/`, `scripts/p2_*`, `tests/p2_*` |
| `p3-queue-results`| P3    | `src/qdec/queue/`, `src/qdec/results/`, `scripts/p3_*`, `tests/p3_*`, `slides/` |

- NEVER edit `src/qdec/contracts.py`, `src/qdec/mocks.py`, `pyproject.toml` or this file
  on a feature branch. If a contract must change, stop and tell the human; they open a
  small PR to `main`.
- Never import another owner's package. Cross-area data moves ONLY through the files
  and schemas defined in `contracts.py`.
- Until real upstream files exist, develop against `qdec.mocks`.

## Conventions
- Python 3.11, numpy, pandas + pyarrow (parquet), stim, sinter, ldpc, pymatching, matplotlib.
- Every table written to disk must pass `qdec.contracts` schema: `df.astype(<COLS>)`
  and `tests/test_contracts.py`.
- Seeds come from `contracts.SEED_TRAIN` / `SEED_TEST`. Cache is built on TRAIN only.
- Every hardware number is labeled MODELED and has a source in
  `src/qdec/costmodel/op_costs.yaml` (`source:` field with citation).
- Use `QUICK_GRID` while developing; `FULL_GRID` only for the Day-3 sweep.
- Before pushing: `make test` must pass.
