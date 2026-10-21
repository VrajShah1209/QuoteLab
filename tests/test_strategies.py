import math

import numpy as np
import pytest

from quotelab.market import LB, MB, MarketConfig
from quotelab.signals import HawkesFlowFeature
from quotelab.simulator import MarketView
from quotelab.book import OrderBook
from quotelab.strategies import (GLFT, AvellanedaStoikov, glft_coefficients, hamiltonian,
                                 tick_quotes)


def view(bid=9999, ask=10000, qb=10, qa=10):
    return MarketView(OrderBook(), bid, ask, qb, qa)


def test_glft_small_gamma_limit():
    half, skew = glft_coefficients(1e-8, 0.7, 1.2, 2.0)
    assert half - skew / 2 == pytest.approx(1 / 2.0, rel=1e-6)  # markup c1 -> 1/k
    assert skew < 1e-3                                           # inventory skew -> 0


def test_glft_skew_increases_with_risk_aversion_and_vol():
    s1 = glft_coefficients(0.05, 0.7, 1.2, 2.0)[1]
    s2 = glft_coefficients(0.5, 0.7, 1.2, 2.0)[1]
    s3 = glft_coefficients(0.05, 1.4, 1.2, 2.0)[1]
    assert s2 > s1 and s3 == pytest.approx(2 * s1)


@pytest.mark.parametrize("fair", [10000.0, 10000.5, 10000.3, 9999.81])
@pytest.mark.parametrize("q", [-4, -1, 0, 2, 5])
def test_hamiltonian_rounding_is_optimal_over_the_whole_grid(fair, q):
    xi, k, c2 = 0.2, 2.0, 0.3
    eta_b, eta_a = (q + 0.5) * c2, (0.5 - q) * c2
    bid, ask = tick_quotes(fair, eta_b, eta_a, xi, k)
    # Brute force over every tick. Note the optimum may sit on the "wrong" side of the fair price
    # when inventory is large (GLFT allows negative distances); the simulator's post-only slide
    # then turns that into joining the touch.
    cand = range(int(fair) - 12, int(fair) + 13)
    best_bid = max(cand, key=lambda p: hamiltonian(fair - p, eta_b, xi, k))
    best_ask = max(cand, key=lambda p: hamiltonian(p - fair, eta_a, xi, k))
    assert (bid, ask) == (best_bid, best_ask)


def test_conservative_rounding_is_never_tighter():
    for fair in np.linspace(10000, 10001, 11):
        h = tick_quotes(fair, 0.1, 0.1, 0.1, 2.0, "hamiltonian")
        c = tick_quotes(fair, 0.1, 0.1, 0.1, 2.0, "conservative")
        assert c[0] <= h[0] and c[1] >= h[1]


def test_inventory_skews_quotes_in_the_right_direction():
    s = GLFT(0.3, 0.7, 1.2, 2.0)
    flat = s.quote(0.0, view(), 0)
    long = s.quote(0.0, view(), 4)
    short = s.quote(0.0, view(), -4)
    assert flat == (9999, 10000)               # join the touch when flat
    assert long[0] < flat[0] and long[1] <= flat[1]
    assert short[1] > flat[1] and short[0] >= flat[0]


def test_avellaneda_stoikov_skew_vanishes_at_the_horizon():
    s = AvellanedaStoikov(0.01, 0.7, 2.0)
    s.t_end = 1000.0
    assert s.inventory_coef(0.0) == pytest.approx(0.01 * 0.49 * 1000.0)
    assert s.inventory_coef(1000.0) == 0.0
    early, late = s.quote(0.0, view(), 3), s.quote(999.9, view(), 3)
    assert early[0] < late[0]  # far from the close, a long position pushes the bid away


def test_alpha_shifts_fair_price():
    from quotelab.signals import AlphaSpec
    s = GLFT(0.1, 0.7, 1.2, 2.0, alpha=AlphaSpec(coef_obi=0.4))
    assert s.fair_price(0.0, view(qb=30, qa=10)) == pytest.approx(9999.5 + 0.4 * 0.5)
    assert s.fair_price(0.0, view(qb=10, qa=30)) == pytest.approx(9999.5 - 0.4 * 0.5)


def test_hawkes_flow_feature_reacts_to_buy_bursts():
    p = MarketConfig().hawkes
    f = HawkesFlowFeature(p, horizon=1.0)
    assert abs(f.value(0.0)) < 1e-9  # symmetric model, empty history
    for t in (0.0, 0.01, 0.02):
        f.on_event(t, MB)
    up = f.value(0.03)
    assert up > 0.1
    assert f.value(30.0) < up / 10  # the forecast decays back as the burst fades
    f.on_event(30.0, LB)
    assert math.isfinite(f.value(30.0))


def test_exact_glft_matches_closed_form_at_low_risk_aversion():
    from quotelab.strategies import glft_exact_eta

    sigma, A, k, Q = 0.7, 1.3, 2.0, 30
    eb, ea = glft_exact_eta(0.03, sigma, A, k, Q)
    c2 = glft_coefficients(0.03, sigma, A, k)[1]
    for q in range(-5, 6):
        assert eb[q + Q] == pytest.approx((q + 0.5) * c2, rel=0.06)
        assert ea[q + Q] == pytest.approx((0.5 - q) * c2, rel=0.06)


def test_exact_glft_is_symmetric_monotone_and_stable():
    from quotelab.strategies import glft_exact_eta

    Q = 30
    eb, ea = glft_exact_eta(1.0, 0.7, 1.3, 2.0, Q)  # tails of the eigenvector underflow here
    assert np.all(np.isfinite(eb[:-1])) and np.all(np.isfinite(ea[1:]))
    assert math.isinf(eb[-1]) and math.isinf(ea[0])
    np.testing.assert_allclose(eb, ea[::-1])
    assert np.all(np.diff(eb[:-1]) > 0)  # the longer you are, the further you back off the bid


def test_exact_ratios_agree_with_dense_eigenvector_when_it_is_accurate():
    from quotelab.strategies import glft_exact_eta

    gamma, sigma, A, k, Q = 0.3, 0.7, 1.3, 2.0, 6
    eb, _ = glft_exact_eta(gamma, sigma, A, k, Q)
    q = np.arange(-Q, Q + 1)
    a = 0.5 * k * gamma * sigma**2
    b = A * (1 + gamma / k) ** (-(1 + k / gamma))
    M = np.diag(a * q**2) - b * (np.eye(q.size, k=1) + np.eye(q.size, k=-1))
    f = np.abs(np.linalg.eigh(M)[1][:, 0])
    np.testing.assert_allclose(eb[:-1], np.log(f[:-1] / f[1:]) / k, rtol=1e-8)


def test_exact_glft_strategy_backs_off_at_the_limit():
    s = GLFT(0.3, 0.7, 1.3, 2.0, exact=True, max_inventory=3)
    bid_full, _ = s.quote(0.0, view(), 3)
    assert bid_full < 9999 - 100  # effectively no bid at the inventory limit
    assert s.quote(0.0, view(), 0) == GLFT(0.3, 0.7, 1.3, 2.0).quote(0.0, view(), 0)


def _slide(bid, ask, v):
    """The simulator's post-only slide: never cross the market."""
    return (None if bid is None else min(bid, v.ask - 1)), (None if ask is None else max(ask, v.bid + 1))


@pytest.mark.parametrize("q", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("bid,ask", [(9999, 10000), (9999, 10001), (9998, 10001)])
def test_markout_glft_reduces_to_glft_without_adverse_selection(q, bid, ask):
    from quotelab.signals import MarkoutModel
    from quotelab.strategies import MarkoutGLFT

    v = view(bid, ask)
    plain = GLFT(0.2, 0.75, 1.3, 1.8)
    mk = MarkoutGLFT(0.2, 0.75, 1.3, 1.8, markout=MarkoutModel(const=(0.0, 0.0)))
    assert _slide(*mk.quote(0.0, v, q), v) == _slide(*plain.quote(0.0, v, q), v)


def test_markout_glft_pulls_quotes_when_every_fill_is_toxic():
    from quotelab.signals import MarkoutModel
    from quotelab.strategies import MarkoutGLFT

    mk = MarkoutGLFT(0.2, 0.75, 1.3, 1.8, markout=MarkoutModel(const=(-5.0, -5.0)))
    assert mk.quote(0.0, view(), 0) == (None, None)


def test_markout_glft_backs_off_when_book_imbalance_is_against_us():
    from quotelab.signals import MarkoutModel
    from quotelab.strategies import MarkoutGLFT

    mk = MarkoutGLFT(0.2, 0.75, 1.3, 1.8, markout=MarkoutModel(const=(-0.25, -1.0), b_obi=0.5))
    calm = mk.quote(0.0, view(qb=10, qa=10), 0)
    ask_heavy = mk.quote(0.0, view(qb=2, qa=40), 0)  # sellers queueing: our bid is about to be run over
    assert calm == (9999, 10000)
    assert ask_heavy[0] is None or ask_heavy[0] < calm[0]
    assert ask_heavy[1] == 10000  # and the ask stays at the touch


def test_hamiltonian_is_safe_for_extreme_inventory_terms():
    assert hamiltonian(0.5, 1e6, 0.1, 2.0) == -math.inf
    s = GLFT(0.3, 0.7, 1.3, 2.0, exact=True, max_inventory=3, queue_premium=1.0)
    v = MarketView(OrderBook(), 9999, 10000, 10, 10, my_bid=9999, my_ask=10000)
    bid, ask = s.quote(0.0, v, 3)  # at the limit, with resting orders: must not overflow
    assert ask == 10000 and bid < 9000
