"""Experiment 2: calibrate the market-making models on training episodes only.

* volatility sigma (ticks / sqrt(s)) and its signature plot,
* fill intensity Lambda(delta) = A exp(-k delta), measured with probe quotes inside the simulator
  (so queue priority, latency and the arbitrageur are priced in),
* unconditional predictability: OLS of future mid moves on the Hawkes flow forecast and book
  imbalance, with Newey-West errors, across horizons,
* fill-conditional markout models: what a fill at the touch / behind the touch is worth given
  the state of the book and of the order flow, with episode-clustered errors. The markout
  horizon is the typical inventory holding period measured by Little's law.

Writes results/calibration.json and docs/figures/fig3_calibration.png.
"""

from __future__ import annotations

import numpy as np
from _common import (DURATION, LATENCY, MUTED, SERIES, SURFACE, TRAIN, WORKERS, load_json, save_figure,
                     save_json, setup_style, step)

from quotelab.calibration import (alpha_dataset, fit_alpha, fit_fill_curve, fit_markout, market_only,
                                  markout_dataset, measure_fill_rates, realized_vol, signature)
from quotelab.experiment import run_batch
from quotelab.hawkes import HawkesParams
from quotelab.market import MarketConfig
from quotelab.simulator import ExecutionConfig

DELTAS = [0.5, 1.5, 2.5, 3.5, 4.5]
FIT_DELTAS = 3                      # fit the exponential on the operating range: touch, +1, +2 ticks
HORIZONS = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
FEATURE_SETS = {"obi": ("obi",), "flow": ("flow",), "both": ("flow", "obi")}
MARKOUT_VARIANTS = {"static": (False, False), "obi": (True, False), "full": (True, True)}
PLOT_HORIZONS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def deciles(x: np.ndarray, y: np.ndarray, n: int = 10) -> dict:
    edges = np.quantile(x, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
    return {"x": [float(x[idx == b].mean()) for b in range(n)],
            "y": [float(y[idx == b].mean()) for b in range(n)],
            "se": [float(y[idx == b].std() / np.sqrt(max((idx == b).sum(), 1))) for b in range(n)]}


def main() -> None:
    setup_style()
    import matplotlib.pyplot as plt

    cfg = MarketConfig()
    exe = ExecutionConfig(latency=LATENCY)
    hk_res = load_json("hawkes.json")
    hawkes = HawkesParams.from_dict(hk_res[f"fit_{hk_res['selected']}"]["params"])
    out: dict = {"hawkes_model": hk_res["selected"], "hawkes": hawkes.to_dict()}

    with step("volatility"):
        runs = market_only(cfg, TRAIN, DURATION)
        out["sigma"] = realized_vol(runs, 1.0)
        out["signature"] = signature(runs)

    with step("fill intensity probes"):
        fr = measure_fill_rates(cfg, TRAIN, DURATION, DELTAS, exe, workers=WORKERS)
        A, k = fit_fill_curve(fr["delta"][:FIT_DELTAS], fr["fills"][:FIT_DELTAS], fr["exposure"][:FIT_DELTAS])
        out["fill_rates"] = fr
        out["A"], out["k"] = A, k
        out["A_all_levels"], out["k_all_levels"] = fit_fill_curve(fr["delta"], fr["fills"], fr["exposure"])

    with step("unconditional predictability of mid moves"):
        regs = {}
        for h in HORIZONS:
            ds = alpha_dataset(cfg, TRAIN, DURATION, hawkes, flow_horizon=h, target_horizons=[h], dt=0.25)
            regs[str(h)] = {name: fit_alpha(ds, h, feats) for name, feats in FEATURE_SETS.items()}
        out["alpha_regressions"] = regs
        h_alpha = max(HORIZONS, key=lambda h: regs[str(h)]["both"]["r2"])
        out["alpha_horizon"] = h_alpha
        out["alpha_specs"] = {"obi": {"coef_obi": regs[str(h_alpha)]["obi"]["coef"]["obi"]}}

    with step("inventory holding period (Little's law) from a reference GLFT"):
        ref = run_batch(cfg, [{"kind": "glft", "gamma": 0.1, "sigma": out["sigma"], "A": A, "k": k,
                               "label": "ref"}], TRAIN, DURATION, exe, workers=WORKERS)
        fills_per_s = np.mean([r["n_fills"] for r in ref]) / DURATION
        holding = float(np.mean([r["mean_abs_inventory"] for r in ref]) / fills_per_s)
        out["holding_period"] = holding
        h_mk = float(min([0.5, 1.0, 2.0, 5.0], key=lambda h: abs(np.log(h / holding))))
        out["markout_horizon"] = h_mk

    with step(f"fill-conditional markout models (h = {h_mk:g}s)"):
        ds = markout_dataset(cfg, TRAIN, DURATION, hawkes, h_mk, exe, extra_horizons=PLOT_HORIZONS)
        fits = {name: fit_markout(ds, hawkes, use_obi=uo, use_flow=uf) for name, (uo, uf) in MARKOUT_VARIANTS.items()}
        out["markout"] = fits
        # a second horizon, for the robustness check in experiment 5
        h2 = 2.0 if h_mk != 2.0 else 5.0
        ds2 = markout_dataset(cfg, TRAIN, DURATION, hawkes, h2, exe)
        out["markout_alt_horizon"] = {"horizon": h2, "full": fit_markout(ds2, hawkes, True, True)}
        touch = ds["behind"] <= 0
        out["markout_by_queue"] = {
            "horizons": list(PLOT_HORIZONS),
            "touch": [float(ds[f"markout_{h:g}"][touch].mean()) for h in PLOT_HORIZONS],
            "behind": [float(ds[f"markout_{h:g}"][~touch].mean()) for h in PLOT_HORIZONS],
            "touch_se": [float(ds[f"markout_{h:g}"][touch].std() / np.sqrt(touch.sum())) for h in PLOT_HORIZONS],
            "behind_se": [float(ds[f"markout_{h:g}"][~touch].std() / np.sqrt((~touch).sum())) for h in PLOT_HORIZONS],
        }
        out["markout_deciles"] = {"obi": deciles(ds["obi"][touch], ds["markout"][touch]),
                                  "same_flow": deciles(ds["same"][touch], ds["markout"][touch])}
        c = fits["full"]["coef"]
        pred = (np.where(touch, c["const_touch"], c["const_behind"]) + c["b_same"] * ds["same"]
                + c["b_opp"] * ds["opp"] + c["b_obi"] * ds["obi"])
        out["markout_calibration"] = deciles(pred, ds["markout"])

    save_json("calibration.json", out)

    # ------------------------------------------------------------------ figure 3
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.0))
    a = ax[0, 0]
    d = np.array(fr["delta"])
    se = np.sqrt(np.array(fr["fills"])) / np.array(fr["exposure"])
    xs = np.linspace(0.3, 4.7, 100)
    a.plot(xs, A * np.exp(-k * xs), color=SERIES[1], label=f"fit: {A:.2f} exp(-{k:.2f} delta)")
    a.errorbar(d, fr["rate"], yerr=1.96 * se, fmt="o", color=SERIES[0], ms=7, mec=SURFACE, mew=1.5,
               capsize=0, label="measured (95% CI)")
    a.set_yscale("log")
    a.set_title("Fill intensity vs distance from mid")
    a.set_xlabel("quote distance delta (ticks)")
    a.set_ylabel("fills per second per side")
    a.legend(loc="upper right")

    a = ax[0, 1]
    for (name, label), col in zip([("both", "imbalance + Hawkes flow"), ("obi", "imbalance only"),
                                   ("flow", "Hawkes flow only")], SERIES[:3]):
        a.plot(HORIZONS, [100 * regs[str(h)][name]["r2"] for h in HORIZONS], color=col, marker="o", ms=6,
               mec=SURFACE, mew=1.5, label=label)
    a.set_xscale("log")
    a.set_title("Unconditional predictability of mid moves")
    a.set_xlabel("forecast horizon (s)")
    a.set_ylabel("R-squared (%)")
    a.legend(loc="upper right")

    a = ax[0, 2]
    mq = out["markout_by_queue"]
    for key, label, col in [("touch", "order at the touch", SERIES[0]), ("behind", "order behind the touch", SERIES[1])]:
        a.errorbar(mq["horizons"], mq[key], yerr=1.96 * np.array(mq[f"{key}_se"]), color=col, marker="o", ms=6,
                   mec=SURFACE, mew=1.5, capsize=0, label=label)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xscale("log")
    a.set_title("Fill markouts by queue position")
    a.set_xlabel("seconds after the fill")
    a.set_ylabel("mid move, ticks, signed by our side")
    a.legend(loc="lower left")

    for a, key, title, xlabel in [
        (ax[1, 0], "obi", "Touch fills: markout vs book imbalance", "imbalance on our side (decile mean)"),
        (ax[1, 1], "same_flow", "Touch fills: markout vs Hawkes flow forecast",
         "excess expected market orders hitting our side"),
        (ax[1, 2], None, "Markout model calibration (all fills)", "predicted markout (ticks)"),
    ]:
        dd = out["markout_deciles"][key] if key else out["markout_calibration"]
        if key is None:
            lim = [min(dd["x"] + dd["y"]) - 0.1, max(dd["x"] + dd["y"]) + 0.1]
            a.plot(lim, lim, color=MUTED, lw=1.0)
        a.errorbar(dd["x"], dd["y"], yerr=1.96 * np.array(dd["se"]), fmt="o", color=SERIES[0], ms=7, mec=SURFACE,
                   mew=1.5, capsize=0)
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel(f"realised {h_mk:g}s markout (ticks)")
    fig.tight_layout()
    save_figure(fig, "fig3_calibration")
    print(f"sigma={out['sigma']:.3f}  A={A:.3f}  k={k:.3f}  holding={holding:.2f}s  markout h={h_mk:g}s")
    for name, f in fits.items():
        print(f"  markout[{name}] r2={f['r2']:.4f} coef={ {k: round(v, 3) for k, v in f['coef'].items()} }")


if __name__ == "__main__":
    main()
