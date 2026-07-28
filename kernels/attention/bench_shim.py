# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Minimal, triton-free reimplementation of the ``triton.testing`` benchmark API.

FlyDSL has no triton dependency, but the FMHA benchmark harness was written
against ``triton.testing``. This module provides the three pieces that harness
uses, with the same semantics as upstream triton so numbers stay comparable:

- :func:`do_bench` -- event-timed benchmark with an auto-sized iteration count
  derived from millisecond budgets, and an L2/MALL flush between runs.
- :class:`Benchmark` -- benchmark configuration container.
- :func:`perf_report` -- decorator producing a :class:`Mark` with ``.run()``.

Differences from ``triton.testing``, all deliberate:

- No plotting. Upstream renders a matplotlib PNG plus a ``results.html`` index;
  matplotlib is not a FlyDSL dependency, so ``Benchmark``'s plot-styling
  arguments (``styles``, ``xlabel``, ``x_log``, ``y_log``) are accepted for
  config compatibility and ignored. CSV output and the printed table are
  unchanged.
- No ``grad_to_none`` / ``show_plots`` / ``diff_col``: unused by FlyDSL
  benchmarks.

Note ``flydsl.autotune.do_bench`` also exists but is *not* a substitute here:
its ``warmup``/``rep`` are raw iteration counts (not millisecond budgets), it
returns the median rather than the mean, and it does not flush caches between
runs -- so its numbers are not comparable with triton-based measurements.
"""

from __future__ import annotations

import math
import os
import statistics
from typing import Any, Dict, List

import torch

__all__ = ["do_bench", "Benchmark", "Mark", "perf_report"]

# Matches triton's AMD/NVIDIA backends: a buffer large enough to evict the
# last-level cache, zeroed between timed runs so each run starts cold.
_CACHE_FLUSH_BYTES = 256 * 1024 * 1024


def _quantile(a: List[float], q: List[float]) -> List[float]:
    n = len(a)
    a = sorted(a)

    def get_quantile(qi):
        if not (0 <= qi <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = qi * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(qi) for qi in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"unknown return_mode {return_mode!r}")


def do_bench(fn, warmup=25, rep=100, quantiles=None, return_mode="mean"):
    """Benchmark the runtime of ``fn``, in milliseconds.

    ``warmup`` and ``rep`` are *time budgets in milliseconds*, not iteration
    counts: the function is first timed roughly to derive how many warmup and
    timed iterations fit in each budget. This matches ``triton.testing.do_bench``.

    :param fn: callable taking no arguments.
    :param warmup: warmup budget in ms.
    :param rep: measurement budget in ms.
    :param quantiles: if given, return these quantiles of the runtime
        distribution instead of a summary statistic; ``return_mode`` is ignored.
    :param return_mode: one of ``min``/``max``/``mean``/``median``/``all``.
    :return: runtime(s) in ms.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()
    torch.cuda.synchronize()

    cache = torch.empty(_CACHE_FLUSH_BYTES // 4, dtype=torch.int, device="cuda")

    # Estimate the runtime of the function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Compute number of warmup and repeat iterations from the ms budgets.
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    for _ in range(n_warmup):
        fn()

    for i in range(n_repeat):
        # Clear the last-level cache before each run.
        cache.zero_()
        start_event[i].record()
        fn()
        end_event[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


class Benchmark:
    """Configuration for one benchmark sweep, consumed by :func:`perf_report`."""

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = "",
        ylabel: str = "",
        x_log: bool = False,
        y_log: bool = False,
        styles=None,
    ):
        """``xlabel`` / ``x_log`` / ``y_log`` / ``styles`` are accepted for
        compatibility with triton benchmark configs and ignored (no plotting).

        :param x_names: argument names varied along the x axis.
        :param x_vals: values for ``x_names``; a scalar per row, or a
            tuple/list whose length matches ``x_names``.
        :param line_arg: argument name whose values distinguish series.
        :param line_vals: values for ``line_arg``.
        :param line_names: display name per entry of ``line_vals``.
        :param plot_name: basename of the emitted CSV and printed table.
        :param args: keyword arguments held fixed across the sweep.
        :param ylabel: unit suffix used in the result column headers.
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args


class Mark:
    """A benchmark function bound to its :class:`Benchmark` configurations."""

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench: Benchmark, save_path: str, print_data: bool, save_precision=6, **kwargs):
        import pandas as pd

        y_mean_labels = [f"{x} ({bench.ylabel})" for x in bench.line_names]
        y_min_labels = [f"{x}-min ({bench.ylabel})" for x in bench.line_names]
        y_max_labels = [f"{x}-max ({bench.ylabel})" for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean_labels + y_min_labels + y_max_labels)
        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]
            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwargs)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max

        df = df[x_names + y_mean_labels]
        if print_data:
            print(bench.plot_name + ":")
            print(df.to_string())
        if save_path:
            df.to_csv(
                os.path.join(save_path, f"{bench.plot_name}.csv"),
                float_format=f"%.{save_precision}f",
                index=False,
            )
        return df

    def run(self, print_data=False, save_path="", return_df=False, **kwargs):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        result_dfs = [self._run(bench, save_path, print_data, **kwargs) for bench in benchmarks]
        if return_df:
            return result_dfs[0] if has_single_bench else result_dfs
        return None


def perf_report(benchmarks):
    """Mark a function for benchmarking; call ``.run()`` on the result."""
    return lambda fn: Mark(fn, benchmarks)
