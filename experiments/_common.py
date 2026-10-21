"""Shared configuration for the research pipeline: seed splits, paths and figure style.

Seed discipline
---------------
TRAIN  episodes are the only data used to estimate anything (Hawkes model, volatility, fill
       curve, alpha coefficients).
VALID  episodes are used to choose hyper-parameters (risk aversion, queue premium).
TEST   episodes are touched once, by the final comparison, after every choice is frozen.

Set ``QUOTELAB_FAST=1`` for a quick smoke run of the whole pipeline on a few short episodes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # run without installing the package

RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

FAST = os.environ.get("QUOTELAB_FAST") == "1"
DURATION = 600.0 if FAST else 1800.0           # seconds of trading per episode (after warm-up)
TRAIN = list(range(1000, 1004 if FAST else 1020))
VALID = list(range(2000, 2008 if FAST else 2060))
TEST = list(range(3000, 3012 if FAST else 3250))
LATENCY = 0.005                                # 5 ms order-entry latency (base case)
WORKERS = int(os.environ.get("QUOTELAB_WORKERS", os.cpu_count() or 1))


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
    return path


def load_json(name: str):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def _to_builtin(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


@contextmanager
def step(msg: str):
    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] {msg} ...", flush=True)
    yield
    print(f"[{time.strftime('%H:%M:%S')}]   done in {time.perf_counter() - t0:,.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Figure style: one validated categorical palette, solid hairline grids, ink-coloured text.
# Colours follow the entity: a strategy keeps its colour in every figure.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Legend / drawing order. Colours are assigned in this order from the validated palette, so the
# three strategies that appear together most often get the three most distinct slots.
STRATEGY_ORDER = ["mk_full", "glft", "join_bbo", "avellaneda_stoikov", "mk_obi", "mk_static"]
# Reading order for tables: the full model, its ablations, then the baselines.
TABLE_ORDER = ["mk_full", "mk_obi", "mk_static", "glft", "avellaneda_stoikov", "join_bbo"]
STRATEGY_NAMES = {
    "mk_full": "GLFT + markouts (queue, OBI, Hawkes)",
    "mk_obi": "GLFT + markouts (queue, OBI)",
    "mk_static": "GLFT + markouts (queue)",
    "glft": "GLFT",
    "avellaneda_stoikov": "Avellaneda-Stoikov",
    "join_bbo": "Join BBO",
}
SHORT_NAMES = {
    "mk_full": "MK-full", "mk_obi": "MK-OBI", "mk_static": "MK-queue", "glft": "GLFT",
    "avellaneda_stoikov": "Avellaneda-Stoikov", "join_bbo": "Join BBO",
}
STRATEGY_COLORS = dict(zip(STRATEGY_ORDER, SERIES))


def setup_style() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK2,
        "axes.labelsize": 9.5,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 6,
    })


def save_figure(fig, name: str) -> Path:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
