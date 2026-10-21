"""Market-making strategies.

Prices are integer ticks. The simulator slides any quote that would cross the book (post-only).

* ``JoinBBO``             naive benchmark: always join the market's best bid and best ask.
* ``AvellanedaStoikov``   Avellaneda & Stoikov (2008), finite horizon T (or a rolling horizon).
* ``GLFT``                Gueant, Lehalle & Fernandez-Tapia (2013) closed-form asymptotic quotes
                          for the infinite-horizon problem with inventory bounds, optionally centred
                          on an alpha-adjusted fair price (``signals.LinearAlpha``).
* ``MarkoutGLFT``         GLFT's inventory value function, but each candidate tick is valued with
                          the *expected markout* of a fill there (``signals.MarkoutModel``):
                          queue position, book imbalance and Hawkes flow forecasts make some fills
                          toxic, and the quote is pulled from a side when no tick is worth it.
* ``FixedOffset``         probe used to measure fill intensities at fixed distances from the mid.

Tick-constrained quotes
-----------------------
Both models assume fill intensity Lambda(delta) = A exp(-k delta) and CARA utility. Their optimal
distance from the fair price has the form

    delta*_bid(q) = eta_bid(q) + (1/xi) ln(1 + xi/k),     delta*_ask(q) = eta_ask(q) + (1/xi) ln(1 + xi/k),

where eta is the inventory indifference term (AS: (q + 1/2) gamma sigma^2 tau; GLFT: (q + 1/2) c2)
and the second term is the monopoly markup. On a large-tick book delta* usually falls *between*
two feasible prices, and the choice matters: for half-spread 0.57 ticks, rounding away from the
mid puts the quote a full tick behind the touch. Instead of rounding, ``rounding="hamiltonian"``
evaluates the HJB Hamiltonian

    H(delta) = A exp(-k delta) (1 - exp(-xi (delta - eta))) / xi

at the two neighbouring ticks and quotes the better one, the exact optimiser over the tick grid
given the value function. ``"conservative"`` (floor bids / ceil asks) and ``"nearest"`` are kept
for the ablation study.
"""

from __future__ import annotations

import math

import numpy as np

from .hawkes import HawkesForecaster, HawkesState
from .market import MB, MS
from .signals import AlphaSpec, LinearAlpha, MarkoutModel
from .simulator import MarketView, SimContext, Strategy

_EPS = 1e-9


def _gain(delta: float, edge: float, eta: float, xi: float, k: float) -> float:
    """exp(-k delta) (1 - exp(-xi (edge - eta))) / xi, safe for extreme arguments.

    ``eta`` can be huge (the exact GLFT solution at the inventory limit), which would overflow
    ``expm1``; such a quote is worth minus infinity.
    """
    x = -xi * (edge - eta)
    if x > 700.0:
        return -math.inf
    return math.exp(min(-k * delta, 700.0)) * (-math.expm1(x)) / xi


def hamiltonian(delta: float, eta: float, xi: float, k: float) -> float:
    """Expected instantaneous gain rate (up to the factor A) of quoting ``delta`` from fair value."""
    return _gain(delta, delta, eta, xi, k)


def tick_quotes(fair: float, eta_bid: float, eta_ask: float, xi: float, k: float,
                rounding: str = "hamiltonian", keep_bid: int | None = None, keep_ask: int | None = None,
                queue_premium: float = 0.0) -> tuple[int, int]:
    """Map the continuous optimal quotes onto the integer tick grid.

    ``keep_bid`` / ``keep_ask`` are the prices of orders already working. Moving such an order
    forfeits its queue position, so it is kept unless the new price's Hamiltonian beats the old
    one by more than the factor (1 + queue_premium).
    """
    markup = math.log1p(xi / k) / xi
    bid_c = fair - (eta_bid + markup)
    ask_c = fair + (eta_ask + markup)
    if rounding == "conservative":
        return math.floor(bid_c + _EPS), math.ceil(ask_c - _EPS)
    if rounding == "nearest":
        return int(math.floor(bid_c + 0.5)), int(math.ceil(ask_c - 0.5))
    if rounding != "hamiltonian":
        raise ValueError(f"unknown rounding {rounding!r}")
    b_lo = math.floor(bid_c + _EPS)
    if hamiltonian(fair - (b_lo + 1), eta_bid, xi, k) > hamiltonian(fair - b_lo, eta_bid, xi, k):
        b_lo += 1
    a_hi = math.ceil(ask_c - _EPS)
    if hamiltonian((a_hi - 1) - fair, eta_ask, xi, k) > hamiltonian(a_hi - fair, eta_ask, xi, k):
        a_hi -= 1
    if queue_premium > 0.0:
        if keep_bid is not None and keep_bid != b_lo and (1.0 + queue_premium) * hamiltonian(
                fair - keep_bid, eta_bid, xi, k) >= hamiltonian(fair - b_lo, eta_bid, xi, k):
            b_lo = keep_bid
        if keep_ask is not None and keep_ask != a_hi and (1.0 + queue_premium) * hamiltonian(
                keep_ask - fair, eta_ask, xi, k) >= hamiltonian(a_hi - fair, eta_ask, xi, k):
            a_hi = keep_ask
    if a_hi <= b_lo:  # can only happen for extreme alpha; keep a one-tick market around fair value
        b_lo, a_hi = math.floor(fair), math.floor(fair) + 1
    return b_lo, a_hi


class JoinBBO(Strategy):
    name = "join_bbo"

    def quote(self, t: float, view: MarketView, inventory: int) -> tuple[int | None, int | None]:
        return view.bid, view.ask


class FixedOffset(Strategy):
    """Quote at mid -/+ delta (ticks), rounded away from the mid."""

    def __init__(self, delta: float):
        self.delta = float(delta)
        self.name = f"fixed_offset_{delta:g}"

    def quote(self, t: float, view: MarketView, inventory: int) -> tuple[int | None, int | None]:
        m = view.mid
        return math.floor(m - self.delta + _EPS), math.ceil(m + self.delta - _EPS)


class _IndifferenceQuoter(Strategy):
    """Shared machinery: fair price (mid + optional alpha) and tick-constrained quoting."""

    xi: float
    k: float
    rounding: str

    def __init__(self, alpha: AlphaSpec | None, alpha_weight: float):
        self.alpha_spec = alpha
        self.alpha_weight = float(alpha_weight)
        self.alpha = LinearAlpha(alpha) if alpha is not None else None

    def reset(self, ctx: SimContext) -> None:
        if self.alpha is not None:
            self.alpha.reset()

    def on_market_event(self, t: float, etype: int) -> None:
        if self.alpha is not None:
            self.alpha.on_market_event(t, etype)

    def fair_price(self, t: float, view: MarketView) -> float:
        m = view.mid
        if self.alpha is not None:
            m += self.alpha_weight * self.alpha.value(t, view)
        return m

    def inventory_coef(self, t: float) -> float:
        raise NotImplementedError

    queue_premium: float = 0.0

    def etas(self, t: float, inventory: int) -> tuple[float, float]:
        """Inventory indifference terms (eta_bid, eta_ask) in ticks."""
        c = self.inventory_coef(t)
        return (inventory + 0.5) * c, (0.5 - inventory) * c

    def quote(self, t: float, view: MarketView, inventory: int) -> tuple[int | None, int | None]:
        eta_bid, eta_ask = self.etas(t, inventory)
        return tick_quotes(self.fair_price(t, view), eta_bid, eta_ask, self.xi, self.k, self.rounding,
                           view.my_bid, view.my_ask, self.queue_premium)


class AvellanedaStoikov(_IndifferenceQuoter):
    """Avellaneda-Stoikov: eta = (q +/- 1/2) gamma sigma^2 tau with tau = T - t (or a constant)."""

    def __init__(self, gamma: float, sigma: float, k: float, horizon: float | None = None,
                 rounding: str = "hamiltonian", alpha: AlphaSpec | None = None, alpha_weight: float = 1.0,
                 queue_premium: float = 0.0, name: str = "avellaneda_stoikov"):
        super().__init__(alpha, alpha_weight)
        self.queue_premium = float(queue_premium)
        self.gamma, self.sigma, self.k = float(gamma), float(sigma), float(k)
        self.xi = self.gamma
        self.horizon = horizon
        self.rounding = rounding
        self.name = name
        self.t_end = math.inf

    def reset(self, ctx: SimContext) -> None:
        super().reset(ctx)
        self.t_end = ctx.t_end

    def inventory_coef(self, t: float) -> float:
        tau = self.horizon if self.horizon is not None else max(self.t_end - t, 0.0)
        return self.gamma * self.sigma**2 * tau


def glft_coefficients(gamma: float, sigma: float, A: float, k: float, xi: float | None = None) -> tuple[float, float]:
    """(half-spread, per-unit skew) of the GLFT asymptotic solution, in ticks.

    delta_bid(q) = c1 + c2/2 + q c2,   delta_ask(q) = c1 + c2/2 - q c2,
    c1 = (1/xi) ln(1 + xi/k),   c2 = sqrt( gamma sigma^2 / (2 A k) * (1 + xi/k)^(1 + k/xi) ),
    with xi = gamma for the CARA-utility objective (Gueant 2017, ch. 10).
    """
    xi = gamma if xi is None else xi
    c1 = math.log1p(xi / k) / xi
    c2 = math.sqrt(gamma * sigma**2 / (2.0 * A * k) * (1.0 + xi / k) ** (1.0 + k / xi))
    return c1 + 0.5 * c2, c2


def glft_exact_eta(gamma: float, sigma: float, A: float, k: float, max_inventory: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact infinite-horizon inventory terms of the GLFT model with |q| <= Q.

    With u = -exp(-gamma (x + q s)) v_q(t)^(-gamma/k), the HJB reduces to the linear system
    dv/dt = M v, M = a diag(q^2) - b (shift up + shift down), a = k gamma sigma^2 / 2,
    b = A (1 + gamma/k)^-(1 + k/gamma), v(T) = 1. Far from the horizon v is proportional to the
    ground-state eigenvector f of M, so the stationary quotes are

        eta_bid(q) = (1/k) ln(f(q) / f(q+1)),   eta_ask(q) = (1/k) ln(f(q) / f(q-1)),

    and the closed form of ``glft_coefficients`` is the Gaussian approximation f(q) ~ exp(-c q^2).
    The eigenvector's tails underflow for large Q, so the ratios rho(q) = f(q)/f(q-1) are obtained
    from the eigen-equation by the (stable) backward continued fraction
    rho(q) = b / (a q^2 - lambda - b rho(q+1)), rho(Q+1) = 0.
    Returns arrays indexed by q + Q; the side that would breach the limit is +inf.
    """
    Q = int(max_inventory)
    q = np.arange(-Q, Q + 1, dtype=float)
    a = 0.5 * k * gamma * sigma**2
    b = A * (1.0 + gamma / k) ** (-(1.0 + k / gamma))
    M = np.diag(a * q**2) - b * (np.eye(q.size, k=1) + np.eye(q.size, k=-1))
    lam = float(np.linalg.eigvalsh(M)[0])
    rho = np.zeros(Q + 2)  # rho[j] = f(j) / f(j-1) for j = 1..Q, rho[Q+1] = 0
    for j in range(Q, 0, -1):
        rho[j] = b / (a * j * j - lam - b * rho[j + 1])
    log_rho = np.log(rho[1:Q + 1])  # j = 1..Q
    eta_ask = np.full(q.size, np.inf)
    for i, qq in enumerate(range(-Q, Q + 1)):
        if qq >= 1:
            eta_ask[i] = log_rho[qq - 1] / k             # ln f(q)/f(q-1)
        elif qq > -Q:
            eta_ask[i] = -log_rho[-qq] / k               # f(q)/f(q-1) = f(|q|)/f(|q|+1) = 1/rho(|q|+1)
    eta_bid = eta_ask[::-1].copy()                       # symmetry: eta_bid(q) = eta_ask(-q)
    return eta_bid, eta_ask


class GLFT(_IndifferenceQuoter):
    """GLFT market maker, optionally with an alpha-adjusted fair price.

    ``exact=True`` uses the exact stationary solution for |q| <= ``max_inventory`` instead of the
    closed-form (Gaussian) approximation.
    """

    def __init__(self, gamma: float, sigma: float, A: float, k: float,
                 alpha: AlphaSpec | None = None, alpha_weight: float = 1.0,
                 rounding: str = "hamiltonian", queue_premium: float = 0.0,
                 exact: bool = False, max_inventory: int = 10, name: str = "glft"):
        super().__init__(alpha, alpha_weight)
        self.queue_premium = float(queue_premium)
        self.gamma, self.sigma, self.A, self.k = float(gamma), float(sigma), float(A), float(k)
        self.xi = self.gamma
        self.half_spread, self.skew = glft_coefficients(gamma, sigma, A, k)
        self.rounding = rounding
        self.exact = exact
        self.max_inventory = int(max_inventory)
        if exact:
            self._eta_b, self._eta_a = glft_exact_eta(gamma, sigma, A, k, max_inventory)
        self.name = name

    def inventory_coef(self, t: float) -> float:
        return self.skew

    def etas(self, t: float, inventory: int) -> tuple[float, float]:
        if not self.exact:
            return super().etas(t, inventory)
        i = min(max(inventory, -self.max_inventory), self.max_inventory) + self.max_inventory
        big = 1e6  # finite stand-in for "do not quote"; the simulator enforces the hard limit
        return min(self._eta_b[i], big), min(self._eta_a[i], big)


class MarkoutGLFT(GLFT):
    """GLFT inventory control with fill-conditional (adverse-selection-aware) quoting.

    For a candidate bid p the quote is worth, per unit time,

        H(p) = A exp(-k delta) (1 - exp(-xi (delta + M(p, x) - eta_bid(q)))) / xi,   delta = mid - p,

    where M(p, x) < 0 is the expected markout of a fill at p in state x. GLFT's own Hamiltonian
    is the special case M = 0. The quote goes to the best of a handful of candidate ticks
    (improve / join / one to three ticks behind), keeps a resting order unless a new tick beats
    it by the queue premium, and is withdrawn when no tick has positive value.
    """

    def __init__(self, gamma: float, sigma: float, A: float, k: float, markout: MarkoutModel,
                 queue_premium: float = 0.0, exact: bool = False, max_inventory: int = 10,
                 depth: int = 3, name: str = "glft_markout"):
        super().__init__(gamma, sigma, A, k, queue_premium=queue_premium, exact=exact,
                         max_inventory=max_inventory, name=name)
        self.markout = markout
        self.depth = int(depth)
        self._forecaster = HawkesForecaster(markout.hawkes, markout.horizon) if markout.uses_flow else None
        self._state = HawkesState(markout.hawkes) if markout.uses_flow else None
        if self._forecaster is not None:
            self._base = self._forecaster.expected_counts(self._forecaster.y_inf)

    def reset(self, ctx: SimContext) -> None:
        super().reset(ctx)
        if self._state is not None:
            self._state = HawkesState(self.markout.hawkes)

    def on_market_event(self, t: float, etype: int) -> None:
        if self._state is not None:
            self._state.update(t, etype)

    def _value(self, delta: float, edge: float, eta: float) -> float:
        return _gain(delta, edge, eta, self.xi, self.k)

    def _best(self, prices, m: float, sign: int, touch: int, eta: float, extra: float, keep: int | None):
        mk = self.markout
        best_v, best_p = 0.0, None
        for p in prices:                    # candidates never cross the market's opposite best
            delta = sign * (m - p)          # distance from the mid (may be <= 0 when improving a wide spread)
            behind = sign * (touch - p)     # ticks behind the market's best price on our side
            edge = delta + mk.const[0 if behind <= 0 else 1] + extra
            v = self._value(delta, edge, eta)
            if p == keep and v > 0:
                v *= 1.0 + self.queue_premium
            if v > best_v:
                best_v, best_p = v, p
        return best_p

    def quote(self, t: float, view: MarketView, inventory: int) -> tuple[int | None, int | None]:
        mk = self.markout
        m = view.mid
        eta_bid, eta_ask = self.etas(t, inventory)
        e_buy = e_sell = 0.0
        if self._forecaster is not None:  # forecast flow in excess of its stationary mean
            counts = self._forecaster.expected_counts(self._state.state_at(t)) - self._base
            e_buy, e_sell = float(counts[MB]), float(counts[MS])
        obi = view.imbalance
        extra_bid = mk.b_same * e_sell + mk.b_opp * e_buy + mk.b_obi * obi
        extra_ask = mk.b_same * e_buy + mk.b_opp * e_sell - mk.b_obi * obi
        bids = range(view.ask - 1, view.bid - self.depth - 1, -1)
        asks = range(view.bid + 1, view.ask + self.depth + 1)
        bid = self._best(bids, m, +1, view.bid, eta_bid, extra_bid, view.my_bid)
        ask = self._best(asks, m, -1, view.ask, eta_ask, extra_ask, view.my_ask)
        if bid is not None and ask is not None and ask <= bid:
            ask = bid + 1
        return bid, ask
