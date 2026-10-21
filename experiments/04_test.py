"""Experiment 4: one-shot out-of-sample comparison on the held-out test episodes.

Every strategy runs on the *same* event streams (common random numbers), so the per-episode PnL
difference between two strategies is the unit of inference: bootstrap confidence intervals and
paired t-tests on the differences, with Holm correction across comparisons.

Writes results/test.json, results/test_paths.npz and docs/figures/fig5_test.png.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import (DURATION, INK, INK2, LATENCY, MUTED, RESULTS, SHORT_NAMES, STRATEGY_COLORS, STRATEGY_NAMES,
                     STRATEGY_ORDER, SURFACE, TABLE_ORDER, TEST, WORKERS, load_json, save_figure, save_json, setup_style,
                     step)

from quotelab.analytics import bootstrap_ci, bootstrap_sharpe_ci, holm_bonferroni, paired_test
from quotelab.experiment import run_batch
from quotelab.market import MarketConfig
from quotelab.simulator import ExecutionConfig

MARKOUT_H = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
METRICS = ["pnl", "spread_capture", "inventory_pnl", "liquidation_cost", "n_fills", "volume", "edge_per_unit",
           "pnl_per_unit", "mean_abs_inventory", "max_drawdown", "order_to_trade", "uptime", "n_orders",
           "n_cancels", "n_late_cancels", "n_fill_while_cancelling", "realized_spread_5s", "sniped_qty"] + \
          [f"markout_{h:g}s" for h in MARKOUT_H]


def run() -> None:
    cfg = MarketConfig()
    val = load_json("validation.json")
    specs = [val["selected"][fam] for fam in STRATEGY_ORDER]

    with step(f"test: {len(specs)} strategies x {len(TEST)} held-out episodes"):
        rows = run_batch(cfg, specs, TEST, DURATION, ExecutionConfig(latency=LATENCY), workers=WORKERS,
                         keep_paths=True, markout_horizons=MARKOUT_H)

    per = {fam: sorted((r for r in rows if r["label"] == fam), key=lambda r: r["seed"]) for fam in STRATEGY_ORDER}
    pnl = {fam: np.array([r["pnl"] for r in rs]) for fam, rs in per.items()}
    out: dict = {"seeds": TEST, "duration": DURATION, "latency": LATENCY, "configs": specs, "strategies": {},
                 "per_episode_pnl": {fam: v.tolist() for fam, v in pnl.items()}}
    for fam, rs in per.items():
        d = {}
        for key in METRICS:
            v = np.array([r[key] for r in rs], float)
            d[key] = float(np.nanmean(v))
        m, lo, hi = bootstrap_ci(pnl[fam])
        s, slo, shi = bootstrap_sharpe_ci(pnl[fam])
        d.update({"pnl_ci": [lo, hi], "pnl_std": float(pnl[fam].std(ddof=1)), "sharpe": s, "sharpe_ci": [slo, shi],
                  "pnl_p05": float(np.quantile(pnl[fam], 0.05)), "share_losing_episodes": float(np.mean(pnl[fam] < 0))})
        out["strategies"][fam] = d

    comps = {}
    for ref in ("glft", "join_bbo"):
        names = [f for f in STRATEGY_ORDER if f != ref]
        tests = {f: paired_test(pnl[f], pnl[ref]) for f in names}
        adj = holm_bonferroni([tests[f]["p_value"] for f in names])
        for f, p in zip(names, adj):
            tests[f]["p_holm"] = p
        comps[ref] = tests
    out["paired_vs"] = comps
    save_json("test.json", out)
    np.savez_compressed(RESULTS / "test_paths.npz", **{
        f"{fam}_wealth": np.stack([r["_wealth"] for r in rs]) for fam, rs in per.items()},
        **{f"{fam}_inventory": np.stack([r["_inventory"] for r in rs]) for fam, rs in per.items()})
    for fam in TABLE_ORDER:
        d = out["strategies"][fam]
        print(f"  {STRATEGY_NAMES[fam]:38s} PnL {d['pnl']:7.1f} [{d['pnl_ci'][0]:6.1f},{d['pnl_ci'][1]:6.1f}]"
              f"  Sharpe {d['sharpe']:.2f}  fills {d['n_fills']:.0f}  |q| {d['mean_abs_inventory']:.2f}")


def plot() -> None:
    """Figure 5, from results/test.json and results/test_paths.npz only."""
    setup_style()
    import matplotlib.pyplot as plt

    out = load_json("test.json")
    paths = np.load(RESULTS / "test_paths.npz")
    pnl = {fam: np.array(v) for fam, v in out["per_episode_pnl"].items()}
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))
    a = ax[0, 0]
    for fam in STRATEGY_ORDER:
        w = paths[f"{fam}_wealth"].astype(float)
        t = np.arange(w.shape[1]) / 60.0
        a.plot(t, w.mean(0), color=STRATEGY_COLORS[fam], label=SHORT_NAMES[fam])
    a.set_title(f"Mean mark-to-market PnL through the episode ({len(out['seeds'])} test episodes)")
    a.set_xlabel("minutes")
    a.set_ylabel("PnL (ticks x lots)")
    a.legend(loc="upper left", fontsize=8)

    a = ax[0, 1]
    order = TABLE_ORDER[::-1]
    box_kw = dict(widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": SURFACE, "lw": 2},
                  whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    try:  # matplotlib >= 3.10
        bp = a.boxplot([pnl[f] for f in order], orientation="horizontal", **box_kw)
    except TypeError:
        bp = a.boxplot([pnl[f] for f in order], vert=False, **box_kw)
    for patch, fam in zip(bp["boxes"], order):
        patch.set_facecolor(STRATEGY_COLORS[fam])
        patch.set_edgecolor(STRATEGY_COLORS[fam])
    for i, fam in enumerate(order, start=1):
        a.plot(pnl[fam].mean(), i, marker="D", color=INK, ms=5, ls="none")
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_yticks(range(1, len(order) + 1), [SHORT_NAMES[f] for f in order])
    a.set_title("Episode PnL distribution (box = IQR, diamond = mean)")
    a.set_xlabel(f"PnL per {out['duration'] / 60:.0f}-min episode (ticks x lots)")

    a = ax[1, 0]
    comp_cols = {"spread_capture": "#6250d6", "inventory_pnl": "#e34948", "liquidation": MUTED}
    ys = np.arange(len(order))
    S = out["strategies"]
    sc = np.array([S[f]["spread_capture"] for f in order])
    ip = np.array([S[f]["inventory_pnl"] for f in order])
    lq = -np.array([S[f]["liquidation_cost"] for f in order])
    a.barh(ys, sc, height=0.5, color=comp_cols["spread_capture"], label="spread captured")
    a.barh(ys, ip, height=0.5, color=comp_cols["inventory_pnl"], label="inventory PnL (adverse selection)")
    a.barh(ys, lq, left=ip, height=0.5, color=comp_cols["liquidation"], label="closing cost")
    a.plot(sc + ip + lq, ys, marker="o", color=INK, ms=6, ls="none", label="net PnL")
    a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(ys, [SHORT_NAMES[f] for f in order])
    a.set_title("Where the PnL comes from (mean per episode)")
    a.set_xlabel("ticks x lots")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)

    a = ax[1, 1]
    for fam in STRATEGY_ORDER:
        mk = [S[fam][f"markout_{h:g}s"] for h in MARKOUT_H]
        a.plot(MARKOUT_H, mk, color=STRATEGY_COLORS[fam], marker="o", ms=5, mec=SURFACE, mew=1.2,
               label=SHORT_NAMES[fam])
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xscale("log")
    a.set_ylim(None, 0.03)
    a.set_title("Markouts: mid move after our fills (negative = adverse)")
    a.set_xlabel("seconds after the fill")
    a.set_ylabel("ticks per lot, signed by our side")
    a.legend(loc="lower left", fontsize=8, ncol=2)
    fig.text(0.5, 0.005, "MK = GLFT with the fill-conditional markout model (queue position; + book imbalance; "
             "+ Hawkes flow forecasts)", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_figure(fig, "fig5_test")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
