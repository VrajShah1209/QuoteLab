"""Parallel experiment runner.

Strategies are described by plain, picklable dicts ("specs") so batches can be shipped to worker
processes. Work is grouped by seed: each worker draws an episode's event stream once and runs
every requested strategy on it, which is exactly the common-random-numbers design used for the
paired comparisons.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Iterable

import numpy as np

from .analytics import episode_metrics
from .hawkes import HawkesParams
from .market import MarketConfig, generate_stream
from .signals import AlphaSpec, MarkoutModel
from .simulator import ExecutionConfig, Simulator, Strategy
from .strategies import GLFT, AvellanedaStoikov, FixedOffset, JoinBBO, MarkoutGLFT


def build_strategy(spec: dict) -> Strategy | None:
    kind = spec["kind"]
    alpha = None
    if spec.get("alpha"):
        a = spec["alpha"]
        alpha = AlphaSpec(
            coef_flow=a.get("coef_flow", 0.0),
            coef_obi=a.get("coef_obi", 0.0),
            flow_horizon=a.get("flow_horizon", 1.0),
            hawkes=HawkesParams.from_dict(a["hawkes"]) if a.get("hawkes") else None,
        )
    if kind == "none":
        return None
    if kind == "join":
        s: Strategy = JoinBBO()
    elif kind == "probe":
        s = FixedOffset(spec["delta"])
    elif kind == "as":
        s = AvellanedaStoikov(spec["gamma"], spec["sigma"], spec["k"], horizon=spec.get("horizon"),
                              rounding=spec.get("rounding", "hamiltonian"), alpha=alpha,
                              alpha_weight=spec.get("alpha_weight", 1.0),
                              queue_premium=spec.get("queue_premium", 0.0))
    elif kind == "glft":
        s = GLFT(spec["gamma"], spec["sigma"], spec["A"], spec["k"], alpha=alpha,
                 alpha_weight=spec.get("alpha_weight", 1.0), rounding=spec.get("rounding", "hamiltonian"),
                 queue_premium=spec.get("queue_premium", 0.0), exact=spec.get("exact", False),
                 max_inventory=spec.get("max_inventory", 10))
    elif kind == "glft_markout":
        s = MarkoutGLFT(spec["gamma"], spec["sigma"], spec["A"], spec["k"],
                        markout=MarkoutModel.from_dict(spec["markout"]),
                        queue_premium=spec.get("queue_premium", 0.0), exact=spec.get("exact", False),
                        max_inventory=spec.get("max_inventory", 10))
    else:
        raise ValueError(f"unknown strategy kind {kind!r}")
    s.name = spec.get("label", s.name)
    return s


def _run_seed(args: tuple) -> list[dict]:
    market, seed, duration, specs, exe, keep_paths, horizons = args
    stream = generate_stream(market, market.warmup + duration, seed)
    out = []
    for spec in specs:
        exe_i = replace(exe, **spec["exe"]) if spec.get("exe") else exe
        res = Simulator(market, stream, build_strategy(spec), exe_i).run()
        m = episode_metrics(res, horizons)
        m["label"] = spec.get("label", m["strategy"])
        if keep_paths:
            m["_wealth"] = res.grid_wealth.astype(np.float32)
            m["_inventory"] = res.grid_inv.astype(np.int16)
        out.append(m)
    return out


def run_batch(market: MarketConfig, specs: list[dict], seeds: Iterable[int], duration: float,
              execution: ExecutionConfig | None = None, workers: int | None = None,
              keep_paths: bool = False, markout_horizons: tuple = (1.0, 5.0, 30.0)) -> list[dict]:
    """Run every spec on every seed. Returns one metrics dict per (spec, seed)."""
    exe = execution or ExecutionConfig()
    tasks = [(market, int(s), duration, specs, exe, keep_paths, tuple(markout_horizons)) for s in seeds]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        chunks = [_run_seed(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            chunks = list(ex.map(_run_seed, tasks, chunksize=1))
    return [m for chunk in chunks for m in chunk]


def by_label(rows: list[dict], key: str) -> dict[str, np.ndarray]:
    """Collect ``key`` per strategy label, ordered by seed (aligned for paired tests)."""
    out: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        out.setdefault(r["label"], []).append((r["seed"], r[key]))
    return {k: np.array([v for _, v in sorted(vals)]) for k, vals in out.items()}
