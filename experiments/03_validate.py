"""Experiment 3: choose hyper-parameters on validation episodes (the test set stays untouched).

Stage 1  GLFT on a (risk aversion gamma) x (queue premium kappa) grid. kappa is how much better
         a new price must be before a resting order gives up its queue position. kappa* is the
         value with the best median Sharpe across the gamma grid; it is then fixed for every
         model-based strategy (it is a property of the execution layer, not of a pricing model).
Stage 2  For each strategy family, gamma is chosen by maximum validation Sharpe ratio of episode PnL.
         The recommended strategy is the markout family with the best validation Sharpe.

Writes results/validation.json (every configuration + the frozen test configurations) and
docs/figures/fig4_validation.png (risk-return frontiers and the queue-premium study).
"""

from __future__ import annotations

import sys

import numpy as np
from _common import (DURATION, LATENCY, MUTED, STRATEGY_COLORS, STRATEGY_NAMES, STRATEGY_ORDER, SURFACE, VALID,
                     WORKERS, load_json, save_figure, save_json, setup_style, step)

from quotelab.analytics import sharpe
from quotelab.experiment import run_batch
from quotelab.market import MarketConfig
from quotelab.simulator import ExecutionConfig

GAMMAS_STAGE1 = [0.03, 0.1, 0.3, 1.0]
KAPPAS = [0.0, 0.5, 1.0, 2.0]
GAMMAS_GLFT = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
GAMMAS_AS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark blue for ordered gamma values


def summarize(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label in dict.fromkeys(r["label"] for r in rows):
        rs = sorted((r for r in rows if r["label"] == label), key=lambda r: r["seed"])
        pnl = np.array([r["pnl"] for r in rs])
        out[label] = {
            "mean_pnl": float(pnl.mean()), "std_pnl": float(pnl.std(ddof=1)), "sharpe": sharpe(pnl),
            "n_fills": float(np.mean([r["n_fills"] for r in rs])),
            "mean_abs_inventory": float(np.mean([r["mean_abs_inventory"] for r in rs])),
            "orders_per_fill": float(np.mean([r["n_orders"] for r in rs]) / max(np.mean([r["n_fills"] for r in rs]), 1)),
            "markout_5s": float(np.mean([r["markout_5s"] for r in rs])),
            "n_episodes": len(rs),
        }
    return out


def main() -> None:
    cfg = MarketConfig()
    cal = load_json("calibration.json")
    exe = ExecutionConfig(latency=LATENCY)
    base = {"sigma": cal["sigma"], "A": cal["A"], "k": cal["k"]}
    out: dict = {"seeds": VALID, "duration": DURATION}

    with step(f"stage 1: GLFT gamma x kappa grid on {len(VALID)} validation episodes"):
        specs = [dict(base, kind="glft", gamma=g, queue_premium=kp, label=f"glft|g={g}|kappa={kp}")
                 for g in GAMMAS_STAGE1 for kp in KAPPAS]
        s1 = summarize(run_batch(cfg, specs, VALID, DURATION, exe, workers=WORKERS))
        med = {kp: float(np.median([s1[f"glft|g={g}|kappa={kp}"]["sharpe"] for g in GAMMAS_STAGE1])) for kp in KAPPAS}
        kappa = max(KAPPAS, key=lambda kp: med[kp])
        out["stage1"] = s1
        out["stage1_median_sharpe"] = {str(k): v for k, v in med.items()}
        out["queue_premium"] = kappa

    with step(f"stage 2: gamma grids per strategy family (kappa*={kappa})"):
        fam_specs = {
            "glft": [dict(base, kind="glft", gamma=g, queue_premium=kappa) for g in GAMMAS_GLFT],
            "mk_static": [dict(base, kind="glft_markout", gamma=g, queue_premium=kappa,
                               markout=cal["markout"]["static"]["model"]) for g in GAMMAS_GLFT],
            "mk_obi": [dict(base, kind="glft_markout", gamma=g, queue_premium=kappa,
                            markout=cal["markout"]["obi"]["model"]) for g in GAMMAS_GLFT],
            "mk_full": [dict(base, kind="glft_markout", gamma=g, queue_premium=kappa,
                             markout=cal["markout"]["full"]["model"]) for g in GAMMAS_GLFT],
            "avellaneda_stoikov": [{"kind": "as", "gamma": g, "sigma": cal["sigma"], "k": cal["k"],
                                    "queue_premium": kappa} for g in GAMMAS_AS],
            "join_bbo": [{"kind": "join"}],
        }
        specs = []
        for fam, lst in fam_specs.items():
            for s in lst:
                s["label"] = f"{fam}|g={s.get('gamma', '-')}"
                specs.append(s)
        s2 = summarize(run_batch(cfg, specs, VALID, DURATION, exe, workers=WORKERS))
        out["stage2"] = s2
        selected = {}
        for fam, lst in fam_specs.items():
            best = max(lst, key=lambda s: s2[s["label"]]["sharpe"])
            selected[fam] = dict(best, label=fam)
        out["selected"] = selected
        out["recommended"] = max(["mk_static", "mk_obi", "mk_full"],
                                 key=lambda f: s2[f"{f}|g={selected[f]['gamma']}"]["sharpe"])
        out["n_configurations_tried"] = len(specs) + len(GAMMAS_STAGE1) * len(KAPPAS)

    save_json("validation.json", out)
    for fam, s in selected.items():
        key = f"{fam}|g={s.get('gamma', '-')}"
        print(f"  {fam:20s} gamma={s.get('gamma', '-')}  validation Sharpe={s2[key]['sharpe']:.2f}")


def plot() -> None:
    """Figure 4, from results/validation.json only."""
    setup_style()
    import matplotlib.pyplot as plt

    out = load_json("validation.json")
    s1, s2, selected = out["stage1"], out["stage2"], out["selected"]
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    a = ax[0]
    for fam in [f for f in STRATEGY_ORDER if f != "join_bbo"]:
        rows = sorted(((float(k.split("=")[1]), v) for k, v in s2.items() if k.startswith(fam + "|")),
                      key=lambda kv: kv[0])
        a.plot([v["std_pnl"] for _, v in rows], [v["mean_pnl"] for _, v in rows], color=STRATEGY_COLORS[fam],
               marker="o", ms=6, mec=SURFACE, mew=1.5, label=STRATEGY_NAMES[fam])
        sel = s2[f"{fam}|g={selected[fam]['gamma']}"]
        a.plot(sel["std_pnl"], sel["mean_pnl"], marker="o", ms=11, color=STRATEGY_COLORS[fam], mec=SURFACE, mew=2)
    j = s2["join_bbo|g=-"]
    a.plot(j["std_pnl"], j["mean_pnl"], marker="s", ms=9, color=STRATEGY_COLORS["join_bbo"], mec=SURFACE, mew=1.5,
           ls="none", label=STRATEGY_NAMES["join_bbo"])
    a.set_title("Risk-return frontiers across gamma (large marker = selected)")
    a.set_xlabel(f"std of PnL per {out['duration'] / 60:.0f}-min episode (ticks x lots)")
    a.set_ylabel("mean PnL per episode")
    a.legend(loc="lower right", fontsize=8)

    grid = {(float(k.split("|g=")[1].split("|")[0]), float(k.split("kappa=")[1])): v for k, v in s1.items()}
    kappas = sorted({kp for _, kp in grid})
    gammas = sorted({g for g, _ in grid})
    for a, key, title, ylabel in [(ax[1], "sharpe", "Queue premium vs Sharpe (GLFT)", "Sharpe per episode"),
                                  (ax[2], "orders_per_fill", "Queue premium vs message traffic (GLFT)",
                                   "orders sent per fill")]:
        for g, col in zip(gammas, ORDINAL):
            a.plot(kappas, [grid[(g, kp)][key] for kp in kappas], color=col, marker="o", ms=6,
                   mec=SURFACE, mew=1.5, label=f"gamma = {g:g}")
        a.set_title(title)
        a.set_xlabel("queue premium kappa")
        a.set_ylabel(ylabel)
    ax[1].axvline(out["queue_premium"], color=MUTED, lw=0.8)
    ax[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig4_validation")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        main()
    plot()
