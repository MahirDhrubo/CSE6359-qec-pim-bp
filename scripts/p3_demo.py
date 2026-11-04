# OWNER: P3
"""Live demo (and its recorded backup): why decode-latency tails, not means, decide whether a
real-time QEC decoder keeps up, and what DecodeCache + the backlog controller buy.

  Act 1  same mean, different tails (synthetic): backlog depends on the tail
  Act 2  real service-time tails per system for one workload (MODELED)
  Act 3  the queue at a stressful round rate: depth sparklines, p99.9, controller off vs on

Deterministic (fixed seeds, files on disk), and it uses the same pipeline as make results, so
every number matches the figures. --record also saves the transcript as a text backup.
Usage: python scripts/p3_demo.py [--grid quick|full] [--mock] [--tag TAG]
           [--utilization 0.9 | --period-ns NS] [--pause 1.5] [--record PATH]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from qdec.contracts import FULL_GRID, QUICK_GRID, System
from qdec.queue.sim import simulate
from qdec.queue.synthetic import same_mean_lognormal
from qdec.results.pipeline import load_inputs, reference_period, system_inputs
from qdec.results.tables import summary_table

BARS = " ▁▂▃▄▅▆▇█"
SPARK_WIDTH = 60


def sparkline(depth: np.ndarray, vmax: float, width: int = SPARK_WIDTH) -> str:
    """Queue depth over time as one line: max depth per bucket, log-scaled against vmax so rows
    drawn with the same vmax are comparable; ' ' = empty queue, '█' = vmax."""
    d = np.asarray(depth, dtype=np.float64)
    if len(d) == 0:
        return ""
    peaks = [chunk.max() for chunk in np.array_split(d, min(width, len(d)))]
    top = np.log1p(max(vmax, 1.0))
    return "".join(BARS[0] if p <= 0 else
                   BARS[max(1, min(8, int(np.ceil(8 * np.log1p(p) / top))))] for p in peaks)


class Narrator:
    """print() with optional pauses between steps and a transcript for --record."""

    def __init__(self, pause: float):
        self.pause, self.lines = pause, []

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def step(self, title: str) -> None:
        if self.pause:
            time.sleep(self.pause)
        self.say()
        self.say(f"== {title} " + "=" * max(0, 70 - len(title)))


def act1(n: Narrator) -> None:
    n.step("Act 1: same mean, different tails (synthetic)")
    period = 1000.0
    df = same_mean_lognormal(0.9 * period, {"light tail": 0.3, "heavy tail": 1.5}, 20_000, 0)
    sims = {k: simulate(g.sort_values("shot_id")["latency_ns"].to_numpy(), period)
            for k, g in df.groupby("system", sort=False)}
    vmax = max(r.queue_depth.max() for r in sims.values())
    n.say(f"round period T = {period:g} ns; both decoders average 0.9 T per window")
    for k, r in sims.items():
        n.say(f"  {k:11s} p99.9 latency {np.quantile(r.latency_ns, 0.999) / period:6.0f} T   "
              f"peak backlog {r.queue_depth.max():4d} windows")
        n.say(f"  {'':11s} |{sparkline(r.queue_depth, vmax)}|")
    n.say("same mean, ~90x worse tail: the tail decides whether the decoder keeps up")


def act2(n: Narrator, tag: str, svc, note: str) -> None:
    n.step(f"Act 2: decode-latency tails, {tag} ({note})")
    t = summary_table(svc)
    n.say(f"  {'system':22s} {'p50':>10s} {'p99':>12s} {'p99.9':>12s}")
    for row in t.itertuples(index=False):
        n.say(f"  {row.system:22s} {row.p50_ns:8,.0f}ns {row.p99_ns:10,.0f}ns "
              f"{row.p999_ns:10,.0f}ns")
    n.say("  the tail is set by the shots BP cannot finish, which fall back to OSD")


def act3(n: Narrator, tag: str, inputs: dict, period: float, how: str) -> dict:
    n.step(f"Act 3: one window every {period / 1e3:,.1f} us ({how})")
    runs = {}
    for system, (s, kw) in inputs.items():
        if system == System.FULL.value and kw:
            runs["full, controller off"] = simulate(s, period)
            runs["full, controller on"] = simulate(s, period, **kw)
        else:
            runs[system] = simulate(s, period, **kw)
    vmax = max(r.queue_depth.max() for r in runs.values())
    for name, r in runs.items():
        flag = "keeps up" if r.stable else "FALLS BEHIND"
        n.say(f"  {name:22s} p99.9 {np.quantile(r.latency_ns, 0.999) / 1e3:12,.1f} us   "
              f"peak {r.queue_depth.max():6,d} windows   {flag}")
        n.say(f"  {'':22s} |{sparkline(r.queue_depth, vmax)}|")
    return runs


def change(before: float, after: float) -> str:
    """'4.2x lower' / '1.3x HIGHER' for a p99.9 going from before to after."""
    return f"{before / after:5.1f}x lower" if after <= before else f"{after / before:5.1f}x HIGHER"


def takeaway(n: Narrator, runs: dict) -> None:
    n.step("Takeaway")
    p = {k: np.quantile(r.latency_ns, 0.999) for k, r in runs.items()}
    nocache, cache = p.get(System.PIM_NOCACHE.value), p.get(System.PIM_CACHE.value)
    off, on = p.get("full, controller off"), p.get("full, controller on")
    if nocache and cache:
        n.say(f"  DecodeCache:          p99.9 {change(nocache, cache)} than PIM without cache")
    if off and on:   # same pricing on both sides, so this is the controller alone
        n.say(f"  backlog controller:   p99.9 {change(off, on)} than with it off, "
              f"same round rate")
    n.say("  caveat: the controller wins by capping BP early; the effect of that on logical")
    n.say("  error rate is not in the cost model. All hardware numbers are MODELED.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tag", default=None, help="workload tag (default: first with data)")
    ap.add_argument("--utilization", type=float, default=0.9,
                    help="round period = PIM_CACHE mean service / utilization")
    ap.add_argument("--period-ns", type=float, default=None, help="override the round period")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between acts (live: 1.5)")
    ap.add_argument("--record", type=Path, default=None, help="also save the transcript here")
    ap.add_argument("--max-iters", type=int, default=None)
    args = ap.parse_args(argv)

    n = Narrator(args.pause)
    note = "MOCK DATA" if args.mock else "MODELED"
    n.say(f"DecodeCache + PIM-BP demo  |  {note}")
    act1(n)

    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    service, functional, _, iter_costs = load_inputs(grid, args.mock)
    tags = list(dict.fromkeys(service["tag"]))
    tag = args.tag or tags[0]
    if tag not in tags:
        raise SystemExit(f"p3_demo: no data for {tag!r}; have {tags}")
    svc = service[service["tag"] == tag]
    act2(n, tag, svc, note)

    inputs = system_inputs(tag, svc, functional[tag], iter_costs, args.max_iters)
    if args.period_ns:
        period, how = args.period_ns, "--period-ns"
    else:
        period = reference_period(inputs, args.utilization)
        how = f"PIM_CACHE at {args.utilization:.0%} utilization"
    runs = act3(n, tag, inputs, period, how)
    takeaway(n, runs)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text("\n".join(n.lines) + "\n", encoding="utf-8")
        print(f"\ntranscript saved to {args.record}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p3_demo: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
