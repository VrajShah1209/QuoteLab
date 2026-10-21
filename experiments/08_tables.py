"""Experiment 8: render every results table from results/*.json and write them into the docs.

Tables are written to results/tables.md and also substituted into README.md and docs/RESEARCH_REPORT.md between
markers of the form

    <!-- BEGIN table:NAME -->
    ...
    <!-- END table:NAME -->

so the numbers in the documentation are generated, never typed by hand.
"""

from __future__ import annotations

import re

from _common import RESULTS, ROOT, SHORT_NAMES, STRATEGY_NAMES, TABLE_ORDER, load_json


def f(x, nd=1):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build() -> dict[str, str]:
    hk = load_json("hawkes.json")
    cal = load_json("calibration.json")
    val = load_json("validation.json")
    test = load_json("test.json")
    rob = load_json("robustness.json")
    lad = load_json("ladder.json")
    T: dict[str, str] = {}

    sf = hk["stylized_facts"]
    T["stylized"] = table(["Statistic", "Value"], [
        ["Exogenous events per second", f(sf["events_per_second"], 1)],
        ["Market orders per second", f(sf["market_orders_per_second"], 2)],
        ["Share of time the spread is 1 tick", f"{100 * sf['spread_share']['1']:.1f}%"],
        ["Share of time the spread is 3+ ticks", f"{100 * sf['spread_share']['3+']:.1f}%"],
        ["Mean depth at best bid / ask (lots)", f"{sf['depth_bid'][0]:.1f} / {sf['depth_ask'][0]:.1f}"],
        ["Volatility at 1 s / 300 s sampling (ticks per sqrt s)",
         f"{sf['sigma_ticks_per_sqrt_s']['1']:.3f} / {sf['sigma_ticks_per_sqrt_s']['300']:.3f}"],
        ["Excess kurtosis of 1 s mid changes", f(sf["excess_kurtosis_1s"], 1)],
        ["Market-order sign autocorrelation, lags 1 / 10", f"{sf['mo_sign_acf']['1']:.3f} / {sf['mo_sign_acf']['10']:.3f}"],
        ["Coefficient of variation of market-order gaps (Poisson = 1)", f(sf["mo_interarrival_cv"], 2)],
    ])

    rows = []
    for r in hk["model_table"]:
        beta = "-" if r["beta"] is None else ", ".join(f"{b:.3g}" for b in r["beta"])
        rho = "-" if r["spectral_radius"] is None else f(r["spectral_radius"], 3)
        rows.append([r["model"], r["n_params"], f(r["loglik"], 0), f(r["bic"], 0), rho, beta])
    T["hawkes_models"] = table(["Model", "Parameters", "Log-likelihood", "BIC", "Branching ratio", "Decay rates (1/s)"],
                               rows) + f"\n\n{hk['n_events_train']:,} training events. Lower BIC is better."

    types = ["market_buy", "market_sell", "limit_buy", "limit_sell"]
    rows = []
    for model, d in hk["gof_heldout"].items():
        rows.append([model] + [f"{d[t]['ks_stat']:.4f} (p {pval(d[t]['p_value'])})" for t in types])
    rc = hk["recovery"]
    T["gof"] = table(["Model", "Market buys", "Market sells", "Limit buys", "Limit sells"], rows) + (
        f"\n\nKolmogorov-Smirnov statistic of time-rescaled inter-arrival times against Exp(1), on five held-out "
        f"episodes. Estimator check: on {rc['reps']} data sets simulated from known parameters, nominal 95% "
        f"intervals covered the truth {100 * rc['coverage_95']:.1f}% of the time (z-scores: mean "
        f"{rc['mean_z']:+.2f}, sd {rc['std_z']:.2f}).")

    fr = cal["fill_rates"]
    T["calibration"] = table(["Quantity", "Value"], [
        ["Volatility sigma (ticks per sqrt s)", f(cal["sigma"], 3)],
        ["Fill intensity A (fills per s per side)", f(cal["A"], 3)],
        ["Fill decay k (per tick)", f(cal["k"], 3)],
        ["Measured fill rate at the touch / 1 / 2 ticks behind (per s)", " / ".join(f"{r:.4f}" for r in fr["rate"][:3])],
        ["Inventory holding period by Little's law", f"{cal['holding_period']:.2f} s"],
        ["Markout horizon used by the strategies", f"{cal['markout_horizon']:g} s"],
    ])

    names = ["const_touch", "const_behind", "b_same", "b_opp", "b_obi"]
    labels = ["At the touch", "Behind the touch", "Hawkes: flow at our side", "Hawkes: flow at other side",
              "Imbalance (own side)"]
    rows = []
    for var, title in [("static", "Queue position"), ("obi", "+ imbalance"), ("full", "+ Hawkes flow")]:
        m = cal["markout"][var]
        rows.append([title] + [f"{m['coef'][n]:+.3f} ({m['t'][n]:+.1f})" if n in m["coef"] else "-" for n in names]
                    + [f"{100 * m['r2']:.1f}%"])
    T["markout_models"] = table(["Model"] + labels + ["R-squared"], rows) + (
        f"\n\nOLS on {cal['markout']['full']['n']:,} fills of probe quotes in training episodes; "
        f"{cal['markout_horizon']:g}-second markouts in ticks; t-statistics clustered by episode in parentheses. "
        "Hawkes features are expected market orders over the horizon in excess of their stationary mean.")

    rows = []
    for fam in TABLE_ORDER:
        s = val["selected"][fam]
        v = val["stage2"][f"{fam}|g={s.get('gamma', '-')}"]
        rows.append([STRATEGY_NAMES[fam], s.get("gamma", "-"), f(v["mean_pnl"]), f(v["std_pnl"]), f(v["sharpe"], 2)])
    T["validation"] = table(["Strategy", "Selected gamma", "Mean PnL", "Std PnL", "Sharpe"], rows) + (
        f"\n\n{len(val['seeds'])} validation episodes; queue premium kappa = {val['queue_premium']:g} chosen first "
        f"on GLFT; {val['n_configurations_tried']} configurations tried in total.")

    S = test["strategies"]
    best = max(TABLE_ORDER, key=lambda fam: S[fam]["sharpe"])
    rows = []
    for fam in TABLE_ORDER:
        d = S[fam]
        sh = f"**{d['sharpe']:.2f}**" if fam == best else f"{d['sharpe']:.2f}"
        rows.append([STRATEGY_NAMES[fam],
                     f"{sh} [{d['sharpe_ci'][0]:.2f}, {d['sharpe_ci'][1]:.2f}]",
                     f"{f(d['pnl'])} [{f(d['pnl_ci'][0])}, {f(d['pnl_ci'][1])}]",
                     f(d["pnl_std"]), f(d["pnl_p05"], 0), f(d["max_drawdown"], 0), f(d["mean_abs_inventory"], 2)])
    T["test_compact"] = table(["Strategy", "Sharpe per episode [95% CI]", "Mean PnL [95% CI]", "Std PnL",
                               "5th pct PnL", "Mean max drawdown", "Mean abs inventory"], rows)

    rows = []
    for fam in TABLE_ORDER:
        d = S[fam]
        rows.append([STRATEGY_NAMES[fam], f(d["n_fills"], 0), f(d["edge_per_unit"], 3), f(d["markout_1s"], 3),
                     f(d["realized_spread_5s"], 3), f(d["pnl_per_unit"], 3), f(d["order_to_trade"], 2),
                     f(d["sniped_qty"], 1), f"{100 * d['share_losing_episodes']:.1f}%"])
    T["test_detail"] = table(["Strategy", "Fills", "Edge per lot", "Markout 1 s", "Realised spread 5 s",
                              "PnL per lot", "Messages per fill", "Lots sniped", "Losing episodes"], rows) + (
        "\n\nPer-lot quantities in ticks. PnL per lot is also the break-even maker fee.")

    for ref, key in (("glft", "paired_glft"), ("join_bbo", "paired_join")):
        rows = []
        for fam in TABLE_ORDER:
            if fam == ref:
                continue
            t = test["paired_vs"][ref][fam]
            rows.append([STRATEGY_NAMES[fam], f"{t['mean_diff']:+.1f} [{t['ci_low']:+.1f}, {t['ci_high']:+.1f}]",
                         f"{t['t_stat']:+.1f}", pval(t["p_holm"]), f"{100 * t['win_rate']:.0f}%",
                         f"{100 * t['variance_reduction']:.0f}%"])
        T[key] = table(["Strategy", "PnL difference [95% CI]", "t", "p (Holm)", "Episodes won",
                        "Variance removed by pairing"], rows)

    rows = []
    for fam in TABLE_ORDER:
        d = S[fam]
        rows.append([STRATEGY_NAMES[fam], f(d["spread_capture"]), f(d["inventory_pnl"]), f(-d["liquidation_cost"]),
                     f(d["pnl"])])
    T["attribution"] = table(["Strategy", "Spread captured", "Inventory PnL", "Closing cost", "Net PnL"], rows)

    step_names = {"glft": "GLFT", "mk_static": "+ queue position", "mk_obi": "+ book imbalance",
                  "mk_full": "+ Hawkes flow"}
    rows = []
    for r in ["glft", "mk_static", "mk_obi", "mk_full"]:
        d = lad["rungs"][r]
        st = lad["steps"].get(r)
        rows.append([
            step_names[r],
            f"{d['sharpe']['value']:.2f} [{d['sharpe']['ci_low']:.2f}, {d['sharpe']['ci_high']:.2f}]",
            "-" if st is None else f"{st['sharpe_diff']:+.2f} [{st['sharpe_diff_ci'][0]:+.2f}, {st['sharpe_diff_ci'][1]:+.2f}]",
            f"{f(d['pnl']['mean'])} [{f(d['pnl']['ci_low'])}, {f(d['pnl']['ci_high'])}]",
            "-" if st is None else f"{st['mean_diff']:+.1f} [{st['ci_low']:+.1f}, {st['ci_high']:+.1f}]",
            f(d["n_fills"], 0), f(d["markout_1s"], 3), f(d["mean_abs_inventory"], 2),
        ])
    T["ladder"] = table(["Model (gamma = %g)" % lad["gamma"], "Sharpe [95% CI]", "Sharpe gain", "Mean PnL [95% CI]",
                         "PnL gain", "Fills", "Markout 1 s", "Mean abs inventory"], rows) + (
        f"\n\nAll rungs on the same {len(lad['seeds'])} test episodes, same risk aversion and queue premium; "
        "gains are paired differences with bootstrap CIs.")

    fq = load_json("fill_quality.json")
    class_names = {"improving": "improving the spread", "at_touch": "at the touch", "behind": "behind the touch",
                   "sniped": "taken by the news arbitrageur"}
    rows = []
    for fam, classes in fq["strategies"].items():
        for c, d in classes.items():
            rows.append([STRATEGY_NAMES[fam] if c == "improving" else "", class_names[c], f"{100 * d['share']:.1f}%",
                         f(d["edge"], 3), f(d["markout"], 3), f(d["realised"], 3), f(d["per_episode"], 1)])
    T["fill_quality"] = table(["Strategy", "Fill type", "Share of volume", "Edge per lot", "Markout 5 s",
                               "Realised spread 5 s", "Realised spread per episode"], rows) + (
        f"\n\nFirst {len(fq['seeds'])} test episodes. Position is measured against the market's best price on our "
        "side just before the trade, excluding our own orders. Per-lot quantities in ticks.")

    rec = rob["recommended"]
    abl_names = {
        "markouts_on_top_of_glft": f"Markout model ({SHORT_NAMES[rec]}) vs plain GLFT",
        "fill_conditional_vs_unconditional_obi": "Imbalance priced per fill vs used as an unconditional alpha",
        "tick_optimal_vs_conservative_rounding": "Tick-optimal quoting vs floor/ceil rounding (GLFT)",
        "queue_premium_glft": "Queue premium vs none (GLFT)",
        "queue_premium_recommended": f"Queue premium vs none ({SHORT_NAMES[rec]})",
        "news_protection_recommended": f"Pulling quotes on news vs ignoring news ({SHORT_NAMES[rec]})",
        "exact_vs_closed_form_glft": "Exact GLFT solution vs closed form",
        "markout_horizon_alt_vs_base": f"Markout horizon {rob['alt_horizon']:g} s vs {cal['markout_horizon']:g} s (MK-full)",
    }
    rows = []
    for k, label in abl_names.items():
        t = rob["ablations"][k]
        rows.append([label, f"{t['mean_diff']:+.1f} [{t['ci_low']:+.1f}, {t['ci_high']:+.1f}]",
                     f"{t['t_stat']:+.1f}", pval(t["p_value"]), f"{100 * t['win_rate']:.0f}%"])
    T["ablations"] = table(["Comparison", "PnL gain per episode [95% CI]", "t", "p", "Episodes won"], rows) + (
        f"\n\nPaired on the first {len(rob['seeds'])} test episodes.")

    lat = rob["latency"]
    fams = list(lat)
    rows = []
    for x in lat[fams[0]]:
        rows.append([f"{1000 * float(x):g} ms"] + [f"{f(lat[fam][x]['mean'])} ({lat[fam][x]['sniped_qty']:.0f})"
                                                   for fam in fams])
    T["latency"] = table(["Order-entry latency"] + [STRATEGY_NAMES[fam] for fam in fams], rows) + (
        "\n\nMean PnL per episode; in parentheses, lots per episode taken by the news arbitrageur, whose reaction "
        "time is Exponential with mean 5 ms. No recalibration across latencies.")

    st = rob["stress"]
    rows = []
    for regime, label in [("base", "Base market"), ("toxic_flow", "Toxic flow"), ("thin_book", "Thin book")]:
        rows.append([f"{label} (branching ratio {st[regime]['spectral_radius']:.2f})"]
                    + [f"{st[regime][fam]['sharpe']:.2f} ({f(st[regime][fam]['mean_pnl'])})" for fam in fams])
    T["stress"] = table(["Market"] + [STRATEGY_NAMES[fam] for fam in fams], rows) + (
        "\n\nSharpe per episode (mean PnL). Strategies keep their base-market calibration. Toxic flow: market-order "
        "self-excitation up by a third. Thin book: 25% fewer limit orders.")
    return T


MARKER = re.compile(r"(<!-- BEGIN table:(\w+) -->\n)(.*?)(<!-- END table:\2 -->)", re.S)


def update_docs(T: dict[str, str]) -> list[str]:
    changed = []
    for path in [ROOT / "README.md", ROOT / "docs" / "RESEARCH_REPORT.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")

        def sub(m: re.Match) -> str:
            name = m.group(2)
            if name not in T:
                raise KeyError(f"{path.name}: unknown table {name!r}")
            return f"{m.group(1)}{T[name]}\n{m.group(4)}"

        new = MARKER.sub(sub, text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(path.name)
    return changed


def main() -> None:
    T = build()
    md = "\n\n".join(f"### {name}\n\n{body}" for name, body in T.items())
    (RESULTS / "tables.md").write_text(md + "\n", encoding="utf-8")
    changed = update_docs(T)
    print(f"wrote results/tables.md ({len(T)} tables); updated: {', '.join(changed) or 'nothing'}")


if __name__ == "__main__":
    main()
