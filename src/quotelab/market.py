"""Hawkes-driven zero-intelligence order-flow model.

Exogenous order flow
--------------------
Four event types arrive as a 4-variate Hawkes process:

    0 = market buy (MB), 1 = market sell (MS), 2 = limit buy (LB), 3 = limit sell (LS)

The default kernel has *two* time scales (a fast ~0.1 s component and a slow ~3 s one), which
mimics the slowly-decaying kernels estimated on real order books. Strategies are calibrated with
a parsimonious single-exponential model, so model misspecification is built into the study.

Marks
-----
* Limit orders are placed ``d`` ticks away from the *opposite* best quote, with
  P(d) proportional to d^-(1 + tail) on {1, ..., max_distance} (Bouchaud, Mezard & Potters 2002).
  A buy at ``best_ask - 1`` therefore improves the bid whenever the spread is wider than a tick.
* Order sizes are geometric.
* Every background limit order has an exponential lifetime (constant per-order cancellation
  hazard, as in Cont, Stoikov & Talreja 2010). This makes depth mean-revert instead of drifting.

Public news and latency arbitrage
---------------------------------
Zero-intelligence flow alone cannot make latency matter: nobody in it reacts to a stale quote. So
the market also receives public "news" (a Poisson process) that moves the fair value by J ticks.
A fast arbitrageur, whose reaction time is random (exponential, a few ms), then sweeps every
resting order priced on the wrong side of the new value. A market maker that sees the news can
cancel its exposed quotes, but its cancel only counts if it reaches the exchange before the sweep:
the latency race of Budish, Cramton & Shim (2015).

All randomness is drawn up front into an ``EventStream``. Running different strategies on the
same stream gives *common random numbers*: paired comparisons then cancel most of the market
noise, which is what makes strategy differences statistically detectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .hawkes import HawkesParams, simulate

MB, MS, LB, LS = 0, 1, 2, 3
EVENT_NAMES = ("market_buy", "market_sell", "limit_buy", "limit_sell")


def _default_hawkes() -> HawkesParams:
    # Branching matrix G[i, j]: expected type-i children of one type-j event.
    #                 MB    MS    LB    LS      <- source j
    g = np.array([[0.45, 0.03, 0.00, 0.04],   # target MB: herding / order splitting
                  [0.03, 0.45, 0.04, 0.00],   # target MS
                  [0.10, 0.35, 0.30, 0.04],   # target LB: bids replenished after sells
                  [0.35, 0.10, 0.04, 0.30]])  # target LS: asks replenished after buys
    weights = np.array([0.6, 0.4])            # fast / slow share of each excitation
    beta = np.array([8.0, 0.35])              # decay rates (1/s)
    mu = np.array([0.4, 0.4, 6.0, 6.0])
    return HawkesParams(mu, g[:, :, None] * weights[None, None, :], beta)


@dataclass(frozen=True)
class MarketConfig:
    """Parameters of the simulated market. Prices are in integer ticks, time in seconds."""

    hawkes: HawkesParams = field(default_factory=_default_hawkes)
    cancel_rate: float = 0.3         # per-order cancellation hazard of background limit orders
    placement_tail: float = 0.2      # P(d) ~ d^-(1 + tail)
    max_distance: int = 10
    mo_size_mean: float = 3.0
    lo_size_mean: float = 2.0
    init_mid: int = 10_000           # initial best ask; the best bid starts one tick below
    init_levels: int = 10
    init_orders_per_level: int = 4
    tick_size: float = 0.01          # currency per tick, for reporting only
    warmup: float = 120.0            # seconds of book evolution before a strategy may trade
    news_rate: float = 0.05          # public fair-value jumps per second
    news_jump_sizes: tuple[int, ...] = (1, 2)
    news_jump_probs: tuple[float, ...] = (0.7, 0.3)
    sniper_latency_mean: float = 0.005  # arbitrageur reaction time ~ Exponential(mean), seconds

    def with_hawkes(self, hawkes: HawkesParams) -> "MarketConfig":
        return replace(self, hawkes=hawkes)


@dataclass
class EventStream:
    """Pre-drawn exogenous randomness for one episode (identical across strategies)."""

    times: np.ndarray
    types: np.ndarray
    distance: np.ndarray
    size: np.ndarray
    lifetime: np.ndarray
    init_orders: list[tuple[int, int, int, float]]  # (side, price, qty, lifetime)
    horizon: float
    seed: int
    news_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    news_jumps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    news_delays: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(self.times.size)

    def hawkes_sequence(self, t0: float = 0.0, t1: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """(times, types, T) restricted to [t0, t1), re-based to start at zero, for estimation."""
        t1 = self.horizon if t1 is None else t1
        m = (self.times >= t0) & (self.times < t1)
        return self.times[m] - t0, self.types[m], float(t1 - t0)


def placement_pmf(tail: float, max_distance: int) -> np.ndarray:
    d = np.arange(1, max_distance + 1, dtype=float)
    w = d ** -(1.0 + tail)
    return w / w.sum()


def generate_stream(cfg: MarketConfig, horizon: float, seed: int) -> EventStream:
    """Draw the exogenous event stream for an episode of length ``horizon`` (incl. warm-up)."""
    ss = np.random.SeedSequence(seed)
    rng_flow, rng_marks, rng_init, rng_news = (np.random.default_rng(s) for s in ss.spawn(4))
    times, types = simulate(cfg.hawkes, horizon, rng_flow)
    n = times.size
    pmf = placement_pmf(cfg.placement_tail, cfg.max_distance)
    distance = rng_marks.choice(np.arange(1, cfg.max_distance + 1), size=n, p=pmf)
    is_mo = types <= MS
    size = np.where(
        is_mo,
        rng_marks.geometric(1.0 / cfg.mo_size_mean, size=n),
        rng_marks.geometric(1.0 / cfg.lo_size_mean, size=n),
    )
    lifetime = rng_marks.exponential(1.0 / cfg.cancel_rate, size=n)

    init_orders = []
    for lvl in range(cfg.init_levels):
        for side, price in ((0, cfg.init_mid - 1 - lvl), (1, cfg.init_mid + lvl)):
            for _ in range(cfg.init_orders_per_level):
                init_orders.append((side, int(price), int(rng_init.geometric(1.0 / cfg.lo_size_mean)),
                                    float(rng_init.exponential(1.0 / cfg.cancel_rate))))
    n_news = rng_news.poisson(cfg.news_rate * horizon)
    news_times = np.sort(rng_news.uniform(0.0, horizon, size=n_news))
    sizes = rng_news.choice(np.asarray(cfg.news_jump_sizes), size=n_news, p=np.asarray(cfg.news_jump_probs))
    news_jumps = (sizes * rng_news.choice([-1, 1], size=n_news)).astype(np.int64)
    news_delays = rng_news.exponential(cfg.sniper_latency_mean, size=n_news)
    return EventStream(times, types.astype(np.int64), distance.astype(np.int64), size.astype(np.int64),
                       lifetime, init_orders, float(horizon), int(seed), news_times, news_jumps, news_delays)
