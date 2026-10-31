# Team workflow

## Branch model

```
main ──●──────────●──────────●──────────●──   (integrated, always passes `make test`)
        \          \          \
         p1-workload-bp    p2-cache-cost    p3-queue-results
```

- `main` holds the shared files (`contracts.py`, `mocks.py`, `io.py`, `pyproject.toml`,
  `Makefile`, `CLAUDE.md`, `tests/test_contracts.py`) and the integrated work.
- Each person works on one long-lived feature branch off `main` and edits only their own paths
  (see the table in [CLAUDE.md](../CLAUDE.md)). Because the paths don't overlap, rebases are
  almost always conflict-free.
- Branches merge into `main` **every evening**, so nobody is ever more than a day out of date.

## Morning routine (everyone, ~2 min)

```bash
git checkout <your-branch>
git fetch && git rebase origin/main
make install          # only if pyproject.toml changed
make test
```

If upstream real files landed on `main` overnight, regenerate them locally (`make data`,
`make functional`, …) because `data/` is git-ignored.

## Evening merge (in this order: P1 → P2 → P3)

Downstream merges last, so it tests against what upstream just shipped.

```bash
git fetch && git rebase origin/main
make test                               # must pass after rebasing
git push --force-with-lease
# open a PR <your-branch> -> main (or fast-forward merge if the team agrees), then:
git checkout main && git pull && git merge --ff-only <your-branch> && git push
```

P2 waits for P1's merge, rebases, runs `make test` and merges. Then P3 does the same.
Post in the team chat when your merge is done.

## Changing a contract

`contracts.py`, `mocks.py`, `io.py` and `pyproject.toml` are frozen. If one really must change:

1. **Stop.** Don't edit it on your feature branch, and don't work around it.
2. Tell the team what has to change and why.
3. Open a tiny PR straight to `main` that touches only the shared file(s), plus `mocks.py` and
   `tests/test_contracts.py` if the schema changed.
4. All three approve, then merge.
5. Everyone runs `git fetch && git rebase origin/main && make install && make test`.

## Integration milestones

| When | Milestone | Done when |
|------|-----------|-----------|
| **End of Day 2** | First real end-to-end on `QUICK_GRID` | Real P1 functional traces → P2 service tables built from them (not mocks) → P3 latency CDF / p99 / backlog plots made from those tables. `make all GRID=quick` runs with no MOCK. |
| **End of Day 3** | Full results | `FULL_GRID` (surface d5–d13 and gross) through the whole pipeline; all 7 `System` baselines in every figure; ablations done. Numbers freeze on Day 4 morning. |

## Day 4

Morning: run `make all GRID=full` from a fresh clone on one machine to confirm reproducibility,
then commit `results/`. After that, only slides and rehearsals.
