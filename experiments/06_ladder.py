"""Experiment 6: where the gain comes from, at a common risk aversion.

The test comparison in experiment 4 lets every family pick its own gamma, so differences mix
"better pricing" with "different risk appetite". Here every rung uses the same gamma (the one
selected for the full model) and the same queue premium, and only the markout model changes:

    GLFT  ->  + queue-position markouts  ->  + book imbalance  ->  + Hawkes flow forecasts

Each step is a paired comparison on all test episodes (common random numbers).
Writes results/ladder.json and docs/figures/fig7_ladder.png.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import (DURATION, INK, INK2, LATENCY, STRATEGY_COLORS, SURFACE, TEST, WORKERS,
                     load_json, save_figure, save_json, setup_style, step)

from quotelab.analytics import bootstrap_ci, bootstrap_sharpe_ci, paired_test
from quotelab.experiment import by_label, run_batch
from quotelab.market import MarketConfig
from quotelab.simulator import ExecutionConfig

RUNGS = ["glft", "mk_static", "mk_obi", "mk_full"]
STEP_NAMES = {"mk_static": "+ queue position", "mk_obi": "+ book imbalance", "mk_full": "+ Hawkes flow"}


def main() -> None:
    cfg = MarketConfig()
    cal = load_json("calibration.json")
    val = load_json("validation.json")
    gamma = val["selected"]["mk_full"]["gamma"]
    kappa = val["queue_premium"]
    base = {"gamma": gamma, "sigma": cal["sigma"], "A": cal["A"], "k": cal["k"], "queue_premium": kappa}
    specs = [dict(base, kind="glft", label="glft")]
    for name in ["static", "obi", "full"]:
        specs.append(dict(base, kind="glft_markout", markout=cal["markout"][name]["model"], label=f"mk_{name}"))

    with step(f"ladder at gamma={gamma:g}, kappa={kappa:g}: {len(specs)} strategies x {len(TEST)} test episodes"):
        rows = run_batch(cfg, specs, TEST, DURATION, ExecutionConfig(latency=LATENCY), workers=WORKERS)

    keys = ["pnl", "n_fills", "markout_1s", "edge_per_unit", "mean_abs_inventory", "order_to_trade", "max_drawdown"]
    data = {k: by_label(rows, k) for k in keys}
    out: dict = {"gamma": gamma, "queue_premium": kappa, "seeds": TEST, "rungs": {}, "steps": {}}
    for r in RUNGS:
        pnl = data["pnl"][r]
        out["rungs"][r] = {
            "pnl": dict(zip(("mean", "ci_low", "ci_high"), bootstrap_ci(pnl))),
            "sharpe": dict(zip(("value", "ci_low", "ci_high"), bootstrap_sharpe_ci(pnl))),
            **{k: float(np.mean(data[k][r])) for k in keys if k != "pnl"},
        }
    for prev, cur in zip(RUNGS[:-1], RUNGS[1:]):
        out["steps"][cur] = paired_test(data["pnl"][cur], data["pnl"][prev])
        # Sharpe difference with a paired bootstrap (resample episodes jointly)
        rng = np.random.default_rng(0)
        a, b = data["pnl"][cur], data["pnl"][prev]
        idx = rng.integers(0, a.size, size=(10_000, a.size))
        sa = a[idx].mean(1) / a[idx].std(1, ddof=1)
        sb = b[idx].mean(1) / b[idx].std(1, ddof=1)
        d = sa - sb
        out["steps"][cur]["sharpe_diff"] = float(a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1))
        out["steps"][cur]["sharpe_diff_ci"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    out["total_vs_glft"] = paired_test(data["pnl"]["mk_full"], data["pnl"]["glft"])
    save_json("ladder.json", out)

    for r in RUNGS:
        d = out["rungs"][r]
        print(f"  {r:10s} PnL {d['pnl']['mean']:7.1f}  Sharpe {d['sharpe']['value']:.2f}  fills {d['n_fills']:.0f}"
              f"  markout1s {d['markout_1s']:.3f}")
    for r, t in out["steps"].items():
        print(f"  step {STEP_NAMES[r]:18s} dPnL {t['mean_diff']:+6.1f} [{t['ci_low']:+.1f},{t['ci_high']:+.1f}]"
              f"  dSharpe {t['sharpe_diff']:+.2f} [{t['sharpe_diff_ci'][0]:+.2f},{t['sharpe_diff_ci'][1]:+.2f}]")


def plot() -> None:
    """Figure 7, from results/ladder.json only."""
    setup_style()
    import matplotlib.pyplot as plt

    out = load_json("ladder.json")
    gamma = out["gamma"]
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    xs = np.arange(len(RUNGS))
    labels = ["GLFT", "+ queue\nposition", "+ book\nimbalance", "+ Hawkes\nflow"]
    for a, key, title, ylabel in [(ax[0], "pnl", "Mean PnL per episode", "ticks x lots (95% CI)"),
                                  (ax[1], "sharpe", "Sharpe per episode", "mean / std of episode PnL (95% CI)")]:
        for x, r in zip(xs, RUNGS):
            d = out["rungs"][r][key]
            v = d["mean"] if key == "pnl" else d["value"]
            a.bar(x, v, width=0.55, color=STRATEGY_COLORS[r])
            a.plot([x, x], [d["ci_low"], d["ci_high"]], color=INK, lw=1.2)
        a.set_xticks(xs, labels)
        a.set_title(f"{title}, gamma = {gamma:g} for every rung")
        a.set_ylabel(ylabel)
        a.grid(axis="x", visible=False)
    for x, r in zip(xs[1:], RUNGS[1:]):
        t = out["steps"][r]
        ax[0].annotate(f"{t['mean_diff']:+.1f}", (x, out["rungs"][r]["pnl"]["ci_high"]), xytext=(0, 4),
                       textcoords="offset points", ha="center", fontsize=9, color=INK2)
        ax[1].annotate(f"{t['sharpe_diff']:+.2f}", (x, out["rungs"][r]["sharpe"]["ci_high"]), xytext=(0, 4),
                       textcoords="offset points", ha="center", fontsize=9, color=INK2)

    a = ax[2]
    offsets = {"glft": (9, -10, "left"), "mk_static": (9, -10, "left"), "mk_obi": (9, -16, "left"),
               "mk_full": (-9, 4, "right")}
    for r, lab in zip(RUNGS, ["GLFT", "+ queue position", "+ book imbalance", "+ Hawkes flow"]):
        d = out["rungs"][r]
        a.plot(d["n_fills"], d["edge_per_unit"], marker="o", ms=10, color=STRATEGY_COLORS[r], mec=SURFACE, mew=2,
               ls="none")
        dx, dy, ha = offsets[r]
        a.annotate(f"{lab}\n{d['mean_abs_inventory']:.2f} lots avg. inventory", (d["n_fills"], d["edge_per_unit"]),
                   xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2, ha=ha)
    a.set_title("Fewer fills, more edge per lot")
    a.set_xlabel("fills per episode")
    a.set_ylabel("edge per lot at the fill (ticks)")
    a.margins(x=0.6, y=0.3)
    fig.tight_layout()
    save_figure(fig, "fig7_ladder")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        main()
    plot()
