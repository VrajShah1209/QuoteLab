"""Performance attribution and statistics.

Per-episode metrics come from a ``SimResult``; cross-episode inference uses the bootstrap and
paired tests (strategies are run on identical event streams, so per-seed differences are the
right unit of analysis). ``ols_hac`` gives Newey-West standard errors for the alpha regressions,
whose overlapping targets make ordinary standard errors far too small.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from .simulator import SimResult


# ---------------------------------------------------------------------------
# Single-episode metrics
# ---------------------------------------------------------------------------
def mid_at(res: SimResult, t: np.ndarray | float) -> np.ndarray:
    """Mid-price (a right-continuous step function) at time(s) t."""
    idx = np.searchsorted(res.mid_t, np.asarray(t, float), side="right") - 1
    return res.mid_v[np.clip(idx, 0, res.mid_v.size - 1)]


def markouts(res: SimResult, horizons: Sequence[float]) -> dict[float, float]:
    """Quantity-weighted mean of side * (m(t + h) - m(t)) per unit, for fills with t + h <= T.

    Negative values are adverse selection: the price moved against the position just acquired.
    """
    out = {}
    for h in horizons:
        m = res.fill_t + h <= res.t_end
        if not m.any():
            out[h] = float("nan")
            continue
        move = (mid_at(res, res.fill_t[m] + h) - res.fill_mid[m]) * res.fill_side[m]
        out[h] = float(np.average(move, weights=res.fill_qty[m]))
    return out


def max_drawdown(wealth: np.ndarray) -> float:
    if wealth.size == 0:
        return 0.0
    peak = np.maximum.accumulate(wealth)
    return float(np.max(peak - wealth))


def episode_metrics(res: SimResult, markout_horizons: Sequence[float] = (1.0, 5.0, 30.0)) -> dict:
    vol = int(res.fill_qty.sum())
    wealth = np.r_[0.0, res.grid_wealth]
    dt = float(res.grid_t[1] - res.grid_t[0]) if res.grid_t.size > 1 else 1.0
    per_min = np.diff(wealth[:: max(int(round(60.0 / dt)), 1)])
    mk = markouts(res, markout_horizons)
    out = {
        "seed": res.seed,
        "strategy": res.strategy,
        "pnl": res.pnl,
        "mtm_pnl": res.mtm_pnl,
        "spread_capture": res.spread_capture,
        "inventory_pnl": res.inventory_pnl,
        "fees": res.fees,
        "liquidation_cost": res.liquidation_cost,
        "n_fills": int(res.fill_t.size),
        "volume": vol,
        "edge_per_unit": res.spread_capture / vol if vol else float("nan"),
        "pnl_per_unit": res.pnl / vol if vol else float("nan"),
        "mean_abs_inventory": float(np.abs(res.grid_inv).mean()) if res.grid_inv.size else 0.0,
        "max_abs_inventory": int(np.abs(res.grid_inv).max()) if res.grid_inv.size else 0,
        "final_inventory": res.final_inventory,
        "max_drawdown": max_drawdown(wealth),
        "minute_pnl_std": float(per_min.std(ddof=1)) if per_min.size > 1 else float("nan"),
        "uptime": float(res.grid_quoting.mean()) if res.grid_quoting.size else 0.0,
        "n_orders": res.n_orders,
        "n_cancels": res.n_cancels,
        "n_rejects": res.n_rejects,
        "n_late_cancels": res.n_late_cancels,
        "n_fill_while_cancelling": res.n_fill_while_cancelling,
        "order_to_trade": (res.n_orders + res.n_cancels) / max(res.fill_t.size, 1),
        "n_news": res.n_news,
        "sniped_qty": res.sniped_qty,
        "sniped_share": res.sniped_qty / vol if vol else 0.0,
    }
    for h, v in mk.items():
        out[f"markout_{h:g}s"] = v
        out[f"realized_spread_{h:g}s"] = out["edge_per_unit"] + v if vol else float("nan")
    return out


# ---------------------------------------------------------------------------
# Cross-episode statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 10_000,
                 level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap confidence interval."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    if stat is np.mean:
        boots = x[idx].mean(axis=1)
    else:
        boots = np.array([stat(x[i]) for i in idx])
    a = (1.0 - level) / 2.0
    return float(stat(x)), float(np.quantile(boots, a)), float(np.quantile(boots, 1.0 - a))


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else float("nan")


def bootstrap_sharpe_ci(x: np.ndarray, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    xb = x[idx]
    boots = xb.mean(axis=1) / xb.std(axis=1, ddof=1)
    a = (1.0 - level) / 2.0
    return sharpe(x), float(np.quantile(boots, a)), float(np.quantile(boots, 1.0 - a))


def paired_test(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Compare strategy a against b on the same seeds (common random numbers)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    t = stats.ttest_1samp(d, 0.0)
    return {
        "mean_diff": mean,
        "ci_low": lo,
        "ci_high": hi,
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
        "win_rate": float(np.mean(d > 0)),
        "corr": float(np.corrcoef(a, b)[0, 1]),
        # how much the pairing helped: var of the difference vs. var if the runs were independent
        "variance_reduction": float(1.0 - d.var(ddof=1) / (np.var(a, ddof=1) + np.var(b, ddof=1))),
    }


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def ols_hac(X: np.ndarray, y: np.ndarray, lags: int, add_const: bool = True) -> dict:
    """OLS with Newey-West (Bartlett kernel) heteroskedasticity- and autocorrelation-robust errors."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    if add_const:
        X = np.column_stack([np.ones(X.shape[0]), X])
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    xu = X * u[:, None]
    s = xu.T @ xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xu[lag:].T @ xu[:-lag]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (u @ u) / ss_tot if ss_tot > 0 else float("nan")
    naive_se = np.sqrt(np.diag(xtx_inv) * (u @ u) / (n - p))
    return {"beta": beta, "se": se, "t": beta / se, "naive_t": beta / naive_se, "r2": float(r2), "n": n}


def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray, add_const: bool = False) -> dict:
    """OLS with cluster-robust (by episode) standard errors and the usual small-sample factor."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(X.shape[0]), X])
    y = np.asarray(y, float)
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((p, p))
    uniq = np.unique(groups)
    for g in uniq:
        m = groups == g
        sg = X[m].T @ u[m]
        meat += np.outer(sg, sg)
    G = uniq.size
    adj = G / max(G - 1, 1) * (n - 1) / max(n - p, 1)
    cov = adj * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"beta": beta, "se": se, "t": beta / se, "r2": float(1 - (u @ u) / ss_tot), "n": n, "clusters": G}


def describe(values: np.ndarray) -> dict:
    v = np.asarray(values, float)
    mean, lo, hi = bootstrap_ci(v)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "std": float(v.std(ddof=1)), "median": float(np.median(v))}


def fmt(x: float, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.{nd}f}"
