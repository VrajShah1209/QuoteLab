"""Calibration of the market-making models from market data (training episodes only).

* ``realized_vol``           sigma in ticks / sqrt(second) from sampled mid-prices.
* ``measure_fill_rates``     fill intensities of fixed-offset probe quotes, *including* queue
                             priority and latency effects, measured in the simulator itself.
* ``fit_fill_curve``         Poisson maximum likelihood for Lambda(delta) = A exp(-k delta).
* ``alpha_dataset`` / ``fit_alpha``  features (Hawkes flow forecast, book imbalance) and future
                             mid moves on a regular grid; OLS with Newey-West errors.
* ``markout_dataset`` / ``fit_markout``  fills of probe quotes with their pre-trade state and
                             subsequent markouts; OLS with episode-clustered errors -> ``MarkoutModel``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import optimize

from .analytics import ols_cluster, ols_hac
from .experiment import run_batch
from .hawkes import HawkesParams
from .market import MarketConfig, generate_stream
from .analytics import mid_at
from .market import MB, MS
from .signals import MarkoutModel, flow_feature_on_grid, hawkes_counts_at, imbalance
from .simulator import ExecutionConfig, SimResult, Simulator
from .strategies import FixedOffset


def market_only(market: MarketConfig, seeds, duration: float, sample_dt: float = 1.0,
                record_trades: bool = False) -> list[SimResult]:
    out = []
    for s in seeds:
        stream = generate_stream(market, market.warmup + duration, int(s))
        out.append(Simulator(market, stream, None, ExecutionConfig(sample_dt=sample_dt), record_trades).run())
    return out


def realized_vol(results: list[SimResult], lag_seconds: float = 1.0) -> float:
    """sqrt(mean squared mid change over ``lag_seconds`` / lag): ticks per sqrt(second)."""
    sq, n = 0.0, 0
    for r in results:
        dt = r.grid_t[1] - r.grid_t[0]
        step = max(int(round(lag_seconds / dt)), 1)
        d = r.grid_mid[step:] - r.grid_mid[:-step]
        sq += float((d**2).sum())
        n += d.size
    return float(np.sqrt(sq / n / lag_seconds))


def signature(results: list[SimResult], lags=(1, 2, 5, 10, 30, 60, 120, 300)) -> dict[float, float]:
    """Volatility signature plot: realised vol as a function of the sampling interval."""
    return {float(h): realized_vol(results, h) for h in lags}


def measure_fill_rates(market: MarketConfig, seeds, duration: float, deltas,
                       execution: ExecutionConfig, workers: int | None = None) -> dict:
    """Fills per second per side for a probe quoting mid -/+ delta (no inventory limit)."""
    exe = replace(execution, max_inventory=10**9)
    specs = [{"kind": "probe", "delta": float(d), "label": f"probe_{d:g}"} for d in deltas]
    rows = run_batch(market, specs, seeds, duration, exe, workers=workers)
    out = {"delta": [], "fills": [], "exposure": [], "rate": []}
    for d in deltas:
        rs = [r for r in rows if r["label"] == f"probe_{d:g}"]
        fills = sum(r["n_fills"] for r in rs)
        exposure = sum(2.0 * duration * r["uptime"] for r in rs)  # seconds of two-sided quoting
        out["delta"].append(float(d))
        out["fills"].append(int(fills))
        out["exposure"].append(float(exposure))
        out["rate"].append(fills / exposure)
    return out


def fit_fill_curve(delta, fills, exposure) -> tuple[float, float]:
    """Poisson MLE of Lambda(delta) = A exp(-k delta) given fill counts and exposure times."""
    delta, fills, exposure = (np.asarray(x, float) for x in (delta, fills, exposure))

    def nll(p):
        loga, k = p
        lam = np.exp(loga - k * delta)
        return float((lam * exposure - fills * (loga - k * delta)).sum())

    rate = fills / exposure
    k0, la0 = np.polyfit(delta, -np.log(np.maximum(rate, 1e-12)), 1)
    res = optimize.minimize(nll, [-la0, k0], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
    return float(np.exp(res.x[0])), float(res.x[1])


def alpha_dataset(market: MarketConfig, seeds, duration: float, hawkes: HawkesParams,
                  flow_horizon: float, target_horizons, dt: float = 0.25) -> dict:
    """Grid samples of features and future mid moves (in ticks) from market-only episodes."""
    flow, obi, spread = [], [], []
    targets: dict[float, list] = {float(h): [] for h in target_horizons}
    episode = []
    for s in seeds:
        stream = generate_stream(market, market.warmup + duration, int(s))
        r = Simulator(market, stream, None, ExecutionConfig(sample_dt=dt)).run()
        f = flow_feature_on_grid(hawkes, flow_horizon, stream.times, stream.types, r.grid_t)
        hmax = max(targets)
        n_use = r.grid_t.size - int(round(hmax / dt))
        flow.append(f[:n_use])
        obi.append(imbalance(r.grid_qbid, r.grid_qask)[:n_use])
        spread.append(r.grid_spread[:n_use])
        for h in targets:
            step = int(round(h / dt))
            targets[h].append((r.grid_mid[step:step + n_use] - r.grid_mid[:n_use]))
        episode.append(np.full(n_use, int(s)))
    return {
        "flow": np.concatenate(flow),
        "obi": np.concatenate(obi),
        "spread": np.concatenate(spread),
        "targets": {h: np.concatenate(v) for h, v in targets.items()},
        "episode": np.concatenate(episode),
        "dt": dt,
    }


def fit_alpha(ds: dict, horizon: float, features=("flow", "obi")) -> dict:
    """Regress the h-second mid move on the chosen features with Newey-West errors."""
    y = ds["targets"][float(horizon)]
    X = np.column_stack([ds[f] for f in features])
    lags = int(np.ceil(2 * horizon / ds["dt"]))
    res = ols_hac(X, y, lags=lags)
    return {
        "features": list(features),
        "horizon": float(horizon),
        "const": float(res["beta"][0]),
        "coef": {f: float(b) for f, b in zip(features, res["beta"][1:])},
        "t_hac": {f: float(t) for f, t in zip(features, res["t"][1:])},
        "t_naive": {f: float(t) for f, t in zip(features, res["naive_t"][1:])},
        "r2": res["r2"],
        "n": res["n"],
    }


def markout_dataset(market: MarketConfig, seeds, duration: float, hawkes: HawkesParams, horizon: float,
                    execution: ExecutionConfig, deltas=(0.5, 1.5), extra_horizons=()) -> dict:
    """Fills of probe quotes (at and behind the touch) with pre-trade features and markouts.

    Fills taken by the news arbitrageur are excluded: those are handled by quote protection,
    and the markout model prices the regular order flow. ``extra_horizons`` adds columns
    ``markout_<h>`` (for plots); all fills kept satisfy t + max(horizons) <= T.
    """
    exe = replace(execution, max_inventory=10**9)
    hmax = max((horizon, *extra_horizons))
    cols = {k: [] for k in ("markout", "behind", "same", "opp", "obi", "episode", "side", "delta")}
    for h in extra_horizons:
        cols[f"markout_{h:g}"] = []
    for s in seeds:
        stream = generate_stream(market, market.warmup + duration, int(s))
        for d in deltas:
            r = Simulator(market, stream, FixedOffset(d), exe).run()
            ok = (~r.fill_sniped) & (r.fill_t + hmax <= r.t_end) & (r.fill_best_bid > 0) & (r.fill_best_ask > 0)
            t = r.fill_t[ok]
            side = r.fill_side[ok]
            counts = hawkes_counts_at(hawkes, horizon, stream.times, stream.types, t, strict=True)
            e_buy, e_sell = counts[:, MB], counts[:, MS]
            qb, qa = r.fill_qty_bid[ok].astype(float), r.fill_qty_ask[ok].astype(float)
            tot = np.maximum(qb + qa, 1.0)
            px = r.fill_price[ok]
            cols["markout"].append(side * (mid_at(r, t + horizon) - r.fill_mid[ok]))
            for h in extra_horizons:
                cols[f"markout_{h:g}"].append(side * (mid_at(r, t + h) - r.fill_mid[ok]))
            cols["behind"].append(np.where(side > 0, r.fill_best_bid[ok] - px, px - r.fill_best_ask[ok]))
            cols["same"].append(np.where(side > 0, e_sell, e_buy))
            cols["opp"].append(np.where(side > 0, e_buy, e_sell))
            cols["obi"].append(side * (qb - qa) / tot)
            cols["episode"].append(np.full(t.size, int(s)))
            cols["side"].append(side)
            cols["delta"].append(np.full(t.size, float(d)))
    return {k: np.concatenate(v) for k, v in cols.items()} | {"horizon": float(horizon)}


def fit_markout(ds: dict, hawkes: HawkesParams | None, use_obi: bool = True, use_flow: bool = True) -> dict:
    """Fit the fill-conditional markout model (bucket intercepts + common slopes)."""
    at_touch = (ds["behind"] <= 0).astype(float)
    behind = 1.0 - at_touch
    names = ["const_touch", "const_behind"]
    cols = [at_touch, behind]
    if use_flow:
        names += ["b_same", "b_opp"]
        cols += [ds["same"], ds["opp"]]
    if use_obi:
        names += ["b_obi"]
        cols += [ds["obi"]]
    res = ols_cluster(np.column_stack(cols), ds["markout"], ds["episode"])
    coef = dict(zip(names, res["beta"].tolist()))
    model = MarkoutModel(
        const=(coef["const_touch"], coef["const_behind"]),
        b_same=coef.get("b_same", 0.0), b_opp=coef.get("b_opp", 0.0), b_obi=coef.get("b_obi", 0.0),
        horizon=ds["horizon"], hawkes=hawkes if use_flow else None,
    )
    return {"model": model.to_dict(), "coef": coef, "t": dict(zip(names, res["t"].tolist())),
            "r2": res["r2"], "n": res["n"], "n_touch": int(at_touch.sum()), "n_behind": int(behind.sum()),
            "mean_markout_touch": float(ds["markout"][at_touch > 0].mean()),
            "mean_markout_behind": float(ds["markout"][behind > 0].mean())}
