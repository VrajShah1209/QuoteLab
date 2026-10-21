"""Experiment 9: the one-figure summary at the top of the README (out-of-sample risk and return).

Writes docs/figures/fig0_summary.png.
"""

from __future__ import annotations

import numpy as np
from _common import (INK, INK2, MUTED, SHORT_NAMES, STRATEGY_COLORS, SURFACE, TABLE_ORDER, load_json, save_figure,
                     setup_style)

LABEL_OFFSETS = {  # (dx, dy) in points, chosen so that labels do not collide
    "mk_full": (-9, 3), "mk_obi": (8, 2), "mk_static": (9, -12), "glft": (8, 2),
    "avellaneda_stoikov": (8, -12), "join_bbo": (-8, 6),
}


def main() -> None:
    setup_style()
    import matplotlib.pyplot as plt

    test = load_json("test.json")
    cal = load_json("calibration.json")
    S = test["strategies"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0, 0.9]})

    a = ax[0]
    order = TABLE_ORDER[::-1]
    ys = np.arange(len(order))
    for y, fam in zip(ys, order):
        d = S[fam]
        a.barh(y, d["sharpe"], height=0.55, color=STRATEGY_COLORS[fam])
        a.plot(d["sharpe_ci"], [y, y], color=INK, lw=1.2)
        a.text(d["sharpe_ci"][1] + 0.08, y, f"{d['sharpe']:.2f}", va="center", fontsize=9, color=INK2)
    a.set_yticks(ys, [SHORT_NAMES[f] for f in order])
    a.set_xlim(0, max(S[f]["sharpe_ci"][1] for f in order) * 1.15)
    a.set_title(f"Out-of-sample Sharpe per episode ({len(test['seeds'])} test episodes)")
    a.set_xlabel("mean / std of episode PnL (bar = estimate, line = 95% CI)")
    a.grid(axis="y", visible=False)

    a = ax[1]
    for fam in TABLE_ORDER:
        d = S[fam]
        a.errorbar(d["pnl_std"], d["pnl"], yerr=[[d["pnl"] - d["pnl_ci"][0]], [d["pnl_ci"][1] - d["pnl"]]],
                   fmt="o", ms=9, color=STRATEGY_COLORS[fam], mec=SURFACE, mew=2, capsize=0, lw=1.5)
        dx, dy = LABEL_OFFSETS[fam]
        a.annotate(SHORT_NAMES[fam], (d["pnl_std"], d["pnl"]), xytext=(dx, dy), textcoords="offset points",
                   fontsize=8.5, color=INK2, ha="left" if dx > 0 else "right")
    a.set_xlim(0, max(S[f]["pnl_std"] for f in TABLE_ORDER) * 1.15)
    a.set_title("Return vs risk")
    a.set_xlabel("std of episode PnL (ticks x lots)")
    a.set_ylabel("mean episode PnL (95% CI)")

    a = ax[2]
    mq = cal["markout_by_queue"]
    h = mq["horizons"]
    for key, label, col in [("behind", "filled behind the touch", INK), ("touch", "filled at the touch", MUTED)]:
        a.plot(h, mq[key], color=col, marker="o", ms=6, mec=SURFACE, mew=1.5)
        a.annotate(label, (h[0], mq[key][0]), xytext=(0, 10), textcoords="offset points", ha="left",
                   fontsize=8.5, color=INK2)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xscale("log")
    a.set_ylim(min(mq["behind"]) * 1.15, 0.1)
    a.set_title("Why: fills behind the touch are toxic")
    a.set_xlabel("seconds after the fill")
    a.set_ylabel("markout, ticks (negative = adverse)")
    fig.tight_layout()
    save_figure(fig, "fig0_summary")


if __name__ == "__main__":
    main()
