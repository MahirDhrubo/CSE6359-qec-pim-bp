# OWNER: P1
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "p1_benchmark_osd", Path(__file__).resolve().parents[1] / "scripts/p1_benchmark_osd.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_times_only_fallback_with_one_warmup_across_chunks(monkeypatch):
    class BP:
        def decode_batch(self, batch, return_llrs):
            assert return_llrs
            return None, None, ~batch[:, 0], np.zeros_like(batch, dtype=float)

    class Decoder:
        calls = 0

        def decode(self, syndrome, llrs):
            assert syndrome[0]
            self.calls += 1

    ticks = iter([10, 15, 20, 29])
    monkeypatch.setattr(mod.time, "perf_counter_ns", lambda: next(ticks))
    decoder = Decoder()
    ids, ns = mod.time_fallbacks(BP(), decoder, np.array([[False], [True], [True]]), chunk=2)
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_array_equal(ns, [5, 9])
    assert decoder.calls == 3  # one warm-up plus two timed fallback calls
