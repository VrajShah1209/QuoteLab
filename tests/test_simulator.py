import numpy as np
import pytest

from quotelab.market import MarketConfig, generate_stream
from quotelab.simulator import ExecutionConfig, Simulator
from quotelab.strategies import GLFT, AvellanedaStoikov, JoinBBO


@pytest.fixture(scope="module")
def market():
    return MarketConfig()


@pytest.fixture(scope="module")
def stream(market):
    return generate_stream(market, market.warmup + 300.0, seed=42)


def strategies():
    return [JoinBBO(), GLFT(0.1, 0.7, 1.2, 2.0), AvellanedaStoikov(1e-3, 0.7, 2.0)]


@pytest.mark.parametrize("strat", strategies(), ids=lambda s: s.name)
def test_pnl_attribution_identity(market, stream, strat):
    exe = ExecutionConfig(latency=0.003, maker_fee=-0.1, max_inventory=6)  # a rebate
    r = Simulator(market, stream, strat, exe).run()
    assert r.fill_t.size > 50
    assert r.mtm_pnl == pytest.approx(r.spread_capture + r.inventory_pnl - r.fees, abs=1e-6)
    assert r.fees == pytest.approx(-0.1 * r.fill_qty.sum())


@pytest.mark.parametrize("strat", strategies(), ids=lambda s: s.name)
def test_passive_fills_always_earn_at_least_half_a_tick(market, stream, strat):
    r = Simulator(market, stream, strat, ExecutionConfig(latency=0.003)).run()
    edge = (r.fill_mid - r.fill_price) * r.fill_side
    assert np.all(edge >= 0.5 - 1e-12)


def test_latency_is_respected(market, stream):
    lat = 0.25
    r = Simulator(market, stream, JoinBBO(), ExecutionConfig(latency=lat)).run()
    assert r.fill_t.min() >= market.warmup + lat


def test_inventory_limit_with_in_flight_overshoot(market, stream):
    exe = ExecutionConfig(latency=0.01, max_inventory=3, quote_size=1)
    r = Simulator(market, stream, JoinBBO(), exe).run()
    path = np.cumsum(r.fill_side * r.fill_qty)
    assert np.abs(path).max() <= exe.max_inventory + exe.quote_size


def test_determinism_and_common_random_numbers(market, stream):
    a = Simulator(market, stream, GLFT(0.1, 0.7, 1.2, 2.0), ExecutionConfig()).run()
    b = Simulator(market, stream, GLFT(0.1, 0.7, 1.2, 2.0), ExecutionConfig()).run()
    assert a.pnl == b.pnl and np.array_equal(a.fill_t, b.fill_t)
    # a different strategy on the same stream sees the same exogenous flow
    c = Simulator(market, stream, None, ExecutionConfig()).run()
    assert c.n_events == a.n_events


def test_book_invariants_after_episode(market, stream):
    sim = Simulator(market, stream, JoinBBO(), ExecutionConfig())
    sim.run()
    sim.book.check_invariants()


def test_market_without_strategy_has_no_pnl(market, stream):
    r = Simulator(market, stream, None, ExecutionConfig()).run()
    assert r.pnl == 0.0 and r.fill_t.size == 0 and r.n_orders == 0
    assert np.all(r.grid_spread[r.grid_spread > 0] >= 1)


def _news_only_stream(market, jump: int, delay: float):
    """A deterministic episode: the initial book, one far-away limit order at t = 6 s (which
    prompts the market maker to quote), and one news item at t = 10 s."""
    from quotelab.market import LB, EventStream

    init = []
    for lvl in range(5):
        init.append((0, market.init_mid - 1 - lvl, 3, 1e9))
        init.append((1, market.init_mid + lvl, 3, 1e9))
    return EventStream(np.array([6.0]), np.array([LB]), np.array([9]), np.array([1]), np.array([1e9]), init,
                       20.0, 0, np.array([10.0]), np.array([jump]), np.array([delay]))


@pytest.mark.parametrize("latency, protection, expect_sniped", [
    (0.001, True, False),   # we see the news and our cancel beats the arbitrageur
    (0.010, True, True),    # we react, but too slowly: the stale ask is taken
    (0.001, False, True),   # fast, but we do not react to news at all
])
def test_latency_race_against_the_arbitrageur(latency, protection, expect_sniped):
    from dataclasses import replace

    market = replace(MarketConfig(), warmup=5.0)
    stream = _news_only_stream(market, jump=+2, delay=0.003)
    sim = Simulator(market, stream, JoinBBO(), ExecutionConfig(latency=latency, news_protection=protection))
    r = sim.run()
    assert r.n_news == 1
    assert (r.sniped_qty > 0) == expect_sniped
    fair = (market.init_mid - 0.5) + 2
    assert sim.book.best_ask >= fair  # nothing is left below the new fair value
    if expect_sniped:
        assert r.final_inventory == -1 and r.inventory_pnl < 0  # sold just before the price jumped
