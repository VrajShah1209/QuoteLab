"""Experiment 5: sensitivity and ablations on the first 100 test episodes (configs frozen).

* Latency sweep: order-entry latency from 0.5 ms to 100 ms, no recalibration. The arbitrageur's
  reaction time is Exponential(mean 5 ms), so this traces out the latency race.
* Ablations: what each design choice is worth (tick-optimal quoting, queue premium, reacting to
  news, the exact vs closed-form GLFT solution, the alpha signals).
* Regime stress: strategies calibrated on the base market are run, unchanged, on a market with
  more toxic (more self-exciting) market-order flow and on a market with thinner liquidity.

Writes results/robustness.json and docs/figures/fig6_robustness.png.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import (DURATION, INK2, LATENCY, MUTED, SHORT_NAMES, STRATEGY_COLORS, SURFACE, TEST,
                     WORKERS, load_json, save_figure, save_json, setup_style, step)

from quotelab.analytics import bootstrap_ci, paired_test, sharpe
from quotelab.experiment import by_label, run_batch
from quotelab.hawkes import HawkesParams
from quotelab.market import LB, LS, MB, MS, MarketConfig
from quotelab.simulator import ExecutionConfig

SEEDS = TEST[:100]
LATENCIES = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def stressed_markets(cfg: MarketConfig) -> dict[str, MarketConfig]:
    hp = cfg.hawkes
    toxic = hp.alpha.copy()
    toxic[MB, MB] *= 4 / 3   # market buys trigger more market buys (and likewise for sells)
    toxic[MS, MS] *= 4 / 3
    thin = hp.mu.copy()
    thin[[LB, LS]] *= 0.75   # 25% less passive liquidity arriving
    return {
        "base": cfg,
        "toxic_flow": cfg.with_hawkes(HawkesParams(hp.mu, toxic, hp.beta)),
        "thin_book": cfg.with_hawkes(HawkesParams(thin, hp.alpha, hp.beta)),
    }


def main() -> None:
    cfg = MarketConfig()
    val = load_json("validation.json")
    cal = load_json("calibration.json")
    sel = val["selected"]
    rec = val["recommended"]
    SWEEP = [rec, "glft", "join_bbo"]
    out: dict = {"seeds": SEEDS, "recommended": rec}

    with step(f"latency sweep: {len(LATENCIES)} latencies x {len(SWEEP)} strategies x {len(SEEDS)} episodes"):
        specs = [dict(sel[f], label=f"{f}|lat={lat}", exe={"latency": lat}) for f in SWEEP for lat in LATENCIES]
        rows = run_batch(cfg, specs, SEEDS, DURATION, ExecutionConfig(latency=LATENCY), workers=WORKERS)
        pnl = by_label(rows, "pnl")
        sniped = by_label(rows, "sniped_qty")
        out["latency"] = {f: {str(lat): dict(zip(("mean", "ci_low", "ci_high"), bootstrap_ci(pnl[f"{f}|lat={lat}"])),
                                             sharpe=sharpe(pnl[f"{f}|lat={lat}"]),
                                             sniped_qty=float(sniped[f"{f}|lat={lat}"].mean()))
                              for lat in LATENCIES} for f in SWEEP}

    with step("ablations"):
        g, c, mo = sel["glft"], sel[rec], sel["mk_obi"]
        alt = cal["markout_alt_horizon"]
        specs = [
            dict(g, label="glft"),
            dict(g, label="glft_conservative_rounding", rounding="conservative"),
            dict(g, label="glft_no_queue_premium", queue_premium=0.0),
            dict(g, label="glft_exact", exact=True),
            dict(c, label="rec"),
            dict(c, label="rec_no_queue_premium", queue_premium=0.0),
            dict(c, label="rec_no_news_protection", exe={"news_protection": False}),
            dict(mo, label="mk_obi"),
            {**{k: v for k, v in mo.items() if k != "markout"}, "kind": "glft", "label": "glft_obi_unconditional",
             "alpha": cal["alpha_specs"]["obi"]},
            dict(sel["mk_full"], label="mk_full"),
            dict(sel["mk_full"], label="mk_full_alt_horizon", markout=alt["full"]["model"]),
        ]
        rows = run_batch(cfg, specs, SEEDS, DURATION, ExecutionConfig(latency=LATENCY), workers=WORKERS)
        pnl = by_label(rows, "pnl")
        orders = by_label(rows, "n_orders")
        out["alt_horizon"] = alt["horizon"]
        out["ablations"] = {
            "markouts_on_top_of_glft": paired_test(pnl["rec"], pnl["glft"]),
            "fill_conditional_vs_unconditional_obi": paired_test(pnl["mk_obi"], pnl["glft_obi_unconditional"]),
            "tick_optimal_vs_conservative_rounding": paired_test(pnl["glft"], pnl["glft_conservative_rounding"]),
            "queue_premium_glft": paired_test(pnl["glft"], pnl["glft_no_queue_premium"]),
            "queue_premium_recommended": paired_test(pnl["rec"], pnl["rec_no_queue_premium"]),
            "news_protection_recommended": paired_test(pnl["rec"], pnl["rec_no_news_protection"]),
            "exact_vs_closed_form_glft": paired_test(pnl["glft_exact"], pnl["glft"]),
            "markout_horizon_alt_vs_base": paired_test(pnl["mk_full_alt_horizon"], pnl["mk_full"]),
        }
        out["ablation_levels"] = {k: {"mean_pnl": float(v.mean()), "sharpe": sharpe(v),
                                      "orders": float(orders[k].mean())} for k, v in pnl.items()}

    with step("regime stress (no recalibration)"):
        stress = {}
        for name, mkt in stressed_markets(cfg).items():
            specs = [dict(sel[f], label=f) for f in SWEEP]
            rows = run_batch(mkt, specs, SEEDS, DURATION, ExecutionConfig(latency=LATENCY), workers=WORKERS)
            pnl = by_label(rows, "pnl")
            stress[name] = {f: {"mean_pnl": float(pnl[f].mean()), "ci": list(bootstrap_ci(pnl[f])[1:]),
                                "sharpe": sharpe(pnl[f])} for f in SWEEP}
            stress[name]["spectral_radius"] = mkt.hawkes.spectral_radius
        out["stress"] = stress

    save_json("robustness.json", out)


def plot() -> None:
    """Figure 6, from results/robustness.json and results/calibration.json only."""
    setup_style()
    import matplotlib.pyplot as plt

    out = load_json("robustness.json")
    cal = load_json("calibration.json")
    rec = out["recommended"]
    sweep = list(out["latency"])
    lats = list(out["latency"][sweep[0]])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw={"width_ratios": [1, 1.15, 0.9]})
    a = ax[0]
    lat_ms = np.array([float(x) for x in lats]) * 1000
    for f in sweep:
        d = out["latency"][f]
        mean = np.array([d[x]["mean"] for x in lats])
        lo = np.array([d[x]["ci_low"] for x in lats])
        hi = np.array([d[x]["ci_high"] for x in lats])
        a.fill_between(lat_ms, lo, hi, color=STRATEGY_COLORS[f], alpha=0.10, lw=0)
        a.plot(lat_ms, mean, color=STRATEGY_COLORS[f], marker="o", ms=6, mec=SURFACE, mew=1.5, label=SHORT_NAMES[f])
    a.axvline(5.0, color=MUTED, lw=0.8)
    a.annotate("arbitrageur's mean\nreaction time", xy=(5.0, 1.0), xycoords=("data", "axes fraction"),
               xytext=(4, -4), textcoords="offset points", va="top", fontsize=8, color=INK2)
    a.set_xscale("log")
    a.set_title("PnL vs order-entry latency (no recalibration)")
    a.set_xlabel("order-entry latency (ms)")
    a.set_ylabel("mean PnL per episode (95% CI)")
    a.legend(loc="lower left", fontsize=8)

    a = ax[1]
    labels = {"markouts_on_top_of_glft": f"markout model ({SHORT_NAMES[rec]})\nvs plain GLFT",
              "fill_conditional_vs_unconditional_obi": "OBI priced per fill\nvs as an unconditional alpha",
              "tick_optimal_vs_conservative_rounding": "tick-optimal quoting\nvs floor/ceil rounding",
              "queue_premium_glft": "queue premium (GLFT)",
              "queue_premium_recommended": f"queue premium ({SHORT_NAMES[rec]})",
              "news_protection_recommended": f"pulling quotes on news ({SHORT_NAMES[rec]})",
              "exact_vs_closed_form_glft": "exact GLFT solution\nvs closed form",
              "markout_horizon_alt_vs_base": f"markout horizon {out['alt_horizon']:g}s\n"
                                             f"vs {cal['markout_horizon']:g}s (MK-full)"}
    keys = list(labels)
    ys = np.arange(len(keys))[::-1]
    for y, kname in zip(ys, keys):
        t = out["ablations"][kname]
        a.plot([t["ci_low"], t["ci_high"]], [y, y], color=STRATEGY_COLORS[rec], lw=2.5)
        a.plot(t["mean_diff"], y, marker="o", ms=8, color=STRATEGY_COLORS[rec], mec=SURFACE, mew=1.5)
    a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(ys, [labels[k] for k in keys])
    a.set_title("What each design choice is worth")
    a.set_xlabel("paired PnL gain per episode (95% CI)")

    a = ax[2]
    regimes = ["base", "toxic_flow", "thin_book"]
    names = {"base": "base market", "toxic_flow": "toxic flow", "thin_book": "thin book"}
    x = np.arange(len(regimes))
    w = 0.24
    for i, f in enumerate(sweep):
        vals = [out["stress"][r][f]["sharpe"] for r in regimes]
        a.bar(x + (i - 1) * (w + 0.02), vals, width=w, color=STRATEGY_COLORS[f], label=SHORT_NAMES[f])
    a.set_xticks(x, [names[r] for r in regimes])
    a.axhline(0, color=INK2, lw=0.8)
    a.set_title("Regime stress: Sharpe per episode")
    a.set_ylabel("Sharpe (mean / std of episode PnL)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig6_robustness")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        main()
    plot()
