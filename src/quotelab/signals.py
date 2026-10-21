"""Short-horizon price signals used to skew quotes.

* ``HawkesFlowFeature``  E[N_MB(t, t+h] - N_MS(t, t+h] | F_t], the expected net market-order flow
  over the next h seconds under a *fitted* Hawkes model, in closed form (an affine function of the
  online excitation state, so evaluating it costs one length-DK dot product).
* ``imbalance``          (Q_bid - Q_ask) / (Q_bid + Q_ask) at the best quotes.
* ``LinearAlpha``        a fitted linear combination of the two, in ticks of expected mid change.
* ``MarkoutModel``       the *fill-conditional* version: the expected markout of one of our fills
  given where the order sits in the book and the state of the flow. This is what a market maker
  should price (a fill is not a random draw: it happens precisely when flow arrives at our price,
  Glosten-Milgrom style), and it is fitted on fills of probe quotes in training episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hawkes import HawkesForecaster, HawkesParams, HawkesState
from .market import MB, MS


class HawkesFlowFeature:
    """Expected net market-order flow (buys minus sells, in events) over ``horizon`` seconds."""

    def __init__(self, params: HawkesParams, horizon: float):
        self.params = params
        self.horizon = horizon
        f = HawkesForecaster(params, horizon)
        self.w = f.gain[MB] - f.gain[MS]
        self.c = float(f.offset[MB] - f.offset[MS])
        self.state = HawkesState(params)

    def reset(self) -> None:
        self.state = HawkesState(self.params)

    def on_event(self, t: float, etype: int) -> None:
        self.state.update(t, etype)

    def value(self, t: float) -> float:
        return self.c + float(self.w @ self.state.state_at(t))


def imbalance(q_bid: np.ndarray | float, q_ask: np.ndarray | float) -> np.ndarray | float:
    """(Q_bid - Q_ask) / (Q_bid + Q_ask), defined as 0 for an empty top of book."""
    tot = np.asarray(q_bid, float) + np.asarray(q_ask, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(tot > 0, (np.asarray(q_bid, float) - np.asarray(q_ask, float)) / tot, 0.0)
    return float(out) if out.ndim == 0 else out


def flow_feature_on_grid(params: HawkesParams, horizon: float, times: np.ndarray, types: np.ndarray,
                         grid: np.ndarray) -> np.ndarray:
    """Evaluate ``HawkesFlowFeature`` at sorted grid times by replaying an event stream."""
    feat = HawkesFlowFeature(params, horizon)
    out = np.empty(grid.size)
    j, n = 0, times.size
    for g_idx, g in enumerate(grid):
        while j < n and times[j] <= g:
            feat.on_event(float(times[j]), int(types[j]))
            j += 1
        out[g_idx] = feat.value(float(g))
    return out


@dataclass
class AlphaSpec:
    """Coefficients of the linear alpha (ticks of expected mid move per unit of each feature)."""

    coef_flow: float = 0.0
    coef_obi: float = 0.0
    flow_horizon: float = 1.0
    hawkes: HawkesParams | None = None

    def to_dict(self) -> dict:
        return {"coef_flow": self.coef_flow, "coef_obi": self.coef_obi, "flow_horizon": self.flow_horizon,
                "hawkes": None if self.hawkes is None else self.hawkes.to_dict()}


class LinearAlpha:
    """alpha(t) = coef_flow * HawkesFlow(t) + coef_obi * imbalance(t), in ticks."""

    def __init__(self, spec: AlphaSpec):
        self.spec = spec
        self.use_flow = spec.coef_flow != 0.0 and spec.hawkes is not None
        self.flow = HawkesFlowFeature(spec.hawkes, spec.flow_horizon) if self.use_flow else None

    def reset(self) -> None:
        if self.flow is not None:
            self.flow.reset()

    def on_market_event(self, t: float, etype: int) -> None:
        if self.flow is not None:
            self.flow.on_event(t, etype)

    def value(self, t: float, view) -> float:
        """Expected mid move in ticks; ``view`` is a ``simulator.MarketView``."""
        a = 0.0
        if self.flow is not None:
            a += self.spec.coef_flow * self.flow.value(t)
        if self.spec.coef_obi != 0.0:
            a += self.spec.coef_obi * view.imbalance
        return a


def hawkes_counts_at(params: HawkesParams, horizon: float, times: np.ndarray, types: np.ndarray,
                     query: np.ndarray, strict: bool = True, excess: bool = True) -> np.ndarray:
    """E[N_i(t, t + horizon] | events before t] at each (sorted) query time, shape (Q, D).

    With ``excess`` the stationary expectation lambda_inf * horizon is subtracted, so the feature
    is zero in a "normal" state and measures how much flow the recent history predicts on top.
    ``strict`` excludes events exactly at the query time (e.g. the market order that filled us).
    """
    f = HawkesForecaster(params, horizon)
    base = f.expected_counts(f.y_inf) if excess else 0.0
    state = HawkesState(params)
    out = np.empty((query.size, params.dim))
    j, n = 0, times.size
    for qi, tq in enumerate(query):
        while j < n and (times[j] < tq if strict else times[j] <= tq):
            state.update(float(times[j]), int(types[j]))
            j += 1
        out[qi] = f.expected_counts(state.state_at(float(tq))) - base
    return out


@dataclass
class MarkoutModel:
    """E[signed markout over ``horizon`` | our order fills] in ticks (positive = favourable).

        markout = const[bucket] + b_same * E[N_same] + b_opp * E[N_opp] + b_obi * OBI_own

    bucket 0: the order is at (or inside) the market's best price; 1: one or more ticks behind.
    N_same are market orders that hit our side (sells, for a bid), N_opp the other side, both
    forecast over ``horizon`` by the fitted Hawkes model *in excess of their stationary mean*
    (so the intercepts are the markouts in a typical state); OBI_own is the top-of-book imbalance
    oriented so that positive means our side of the book is the heavier one.
    """

    const: tuple[float, float] = (0.0, 0.0)
    b_same: float = 0.0
    b_opp: float = 0.0
    b_obi: float = 0.0
    horizon: float = 2.0
    hawkes: HawkesParams | None = None

    @property
    def uses_flow(self) -> bool:
        return self.hawkes is not None and (self.b_same != 0.0 or self.b_opp != 0.0)

    def to_dict(self) -> dict:
        return {"const": list(self.const), "b_same": self.b_same, "b_opp": self.b_opp, "b_obi": self.b_obi,
                "horizon": self.horizon, "hawkes": None if self.hawkes is None else self.hawkes.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "MarkoutModel":
        return cls(tuple(d["const"]), d.get("b_same", 0.0), d.get("b_opp", 0.0), d.get("b_obi", 0.0),
                   d.get("horizon", 2.0), HawkesParams.from_dict(d["hawkes"]) if d.get("hawkes") else None)
