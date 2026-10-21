"""Event-driven market simulator and latency-aware market-making backtester.

Five time-ordered streams are merged:

1. exogenous Hawkes order flow (pre-drawn in an ``EventStream``),
2. background cancellations (a heap keyed by each resting order's lifetime),
3. public news (fair-value jumps), which the market maker sees immediately,
4. latency-arbitrage sweeps that follow each news item after a random reaction time and take
   every resting order priced on the wrong side of the new fair value,
5. the market maker's own order messages, which reach the exchange ``latency`` seconds after
   the decision that produced them.

The market maker therefore faces the frictions that dominate real market making: it joins the
*back* of the FIFO queue, loses priority whenever it reprices, and when news arrives its cancels
race the arbitrageur's sweep; a slow cancel means the stale quote is picked off.

Accounting
----------
Cash and inventory are in ticks and units. With mid-price m(t), mark-to-market wealth
W = cash + q m obeys the exact decomposition

    W_T - W_0 = sum_fills qty * edge  +  sum_mid-changes q * dm  -  fees
                (spread capture)        (inventory PnL)

where ``edge = m - p`` for a buy at p (``p - m`` for a sell), m measured just before the
aggressing order. Final PnL also pays the cost of sweeping the closing inventory through the
final book. ``tests/test_simulator.py`` checks the identity to machine precision.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .book import ASK, BID, BUY, SELL, OrderBook
from .market import LB, MB, MS, EventStream, MarketConfig

BG, MM = 0, 1  # order owners: background flow, our market maker

_EVENT, _CANCEL, _ACTION, _NEWS, _SWEEP = 0, 1, 2, 3, 4
_NEW, _CXL = 0, 1
_PNEW, _LIVE, _PCXL = 0, 1, 2


@dataclass(frozen=True)
class ExecutionConfig:
    latency: float = 0.005       # order-entry latency in seconds
    quote_size: int = 1
    max_inventory: int = 10
    maker_fee: float = 0.0       # ticks per unit filled passively (negative = rebate)
    taker_fee: float = 0.0       # ticks per unit for the closing sweep
    sample_dt: float = 1.0       # sampling grid for wealth / inventory / book features
    news_protection: bool = True  # on news, immediately cancel quotes the arbitrageur would take
    news_window: float = 1.0     # ... and keep new quotes on the exposed side safe for this long


@dataclass(frozen=True)
class SimContext:
    """What a strategy is told about the episode before it starts."""

    t_start: float
    t_end: float
    latency: float
    quote_size: int
    max_inventory: int
    tick_size: float


class MarketView:
    """The market as the strategy sees it: best quotes and sizes *excluding its own orders*,
    plus the prices of the strategy's own working orders (``None`` if it has none on a side)."""

    __slots__ = ("book", "bid", "ask", "bid_qty", "ask_qty", "my_bid", "my_ask")

    def __init__(self, book: OrderBook, bid: int, ask: int, bid_qty: int, ask_qty: int,
                 my_bid: int | None = None, my_ask: int | None = None):
        self.book, self.bid, self.ask, self.bid_qty, self.ask_qty = book, bid, ask, bid_qty, ask_qty
        self.my_bid, self.my_ask = my_bid, my_ask

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> int:
        return self.ask - self.bid

    @property
    def imbalance(self) -> float:
        tot = self.bid_qty + self.ask_qty
        return 0.0 if tot == 0 else (self.bid_qty - self.ask_qty) / tot


class Strategy:
    """Base class. Prices are integer ticks; return ``None`` to not quote a side.

    ``quote`` is called after every market-data update (exogenous order or background cancel),
    never in reaction to the strategy's own messages, so quoting cannot feed back on itself.
    """

    name = "strategy"

    def reset(self, ctx: SimContext) -> None:  # noqa: B027 - optional hook
        pass

    def on_market_event(self, t: float, etype: int) -> None:  # noqa: B027 - optional hook
        """Called for every exogenous order (MB/MS/LB/LS), including during warm-up."""

    def quote(self, t: float, view: MarketView, inventory: int) -> tuple[int | None, int | None]:
        raise NotImplementedError


class _MyOrder:
    __slots__ = ("price", "qty", "state")

    def __init__(self, price: int, qty: int, state: int):
        self.price, self.qty, self.state = price, qty, state


@dataclass
class SimResult:
    seed: int
    strategy: str
    t_start: float
    t_end: float
    mid_t: np.ndarray
    mid_v: np.ndarray
    grid_t: np.ndarray
    grid_mid: np.ndarray
    grid_spread: np.ndarray
    grid_qbid: np.ndarray
    grid_qask: np.ndarray
    grid_cash: np.ndarray
    grid_inv: np.ndarray
    grid_quoting: np.ndarray
    fill_t: np.ndarray
    fill_side: np.ndarray
    fill_price: np.ndarray
    fill_qty: np.ndarray
    fill_mid: np.ndarray
    spread_capture: float
    inventory_pnl: float
    fees: float
    liquidation_cost: float
    final_cash: float
    final_inventory: int
    final_mid: float
    n_events: int
    n_orders: int
    n_cancels: int
    n_rejects: int
    n_late_cancels: int
    n_fill_while_cancelling: int
    trades_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trades_sign: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_news: int = 0
    sniped_qty: int = 0
    fill_best_bid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fill_best_ask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fill_qty_bid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fill_qty_ask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fill_sniped: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def mtm_pnl(self) -> float:
        return self.final_cash + self.final_inventory * self.final_mid

    @property
    def pnl(self) -> float:
        return self.mtm_pnl - self.liquidation_cost

    @property
    def grid_wealth(self) -> np.ndarray:
        return self.grid_cash + self.grid_inv * self.grid_mid


class Simulator:
    """Runs one episode of the market, optionally with a market-making strategy."""

    def __init__(
        self,
        market: MarketConfig,
        stream: EventStream,
        strategy: Strategy | None = None,
        execution: ExecutionConfig | None = None,
        record_trades: bool = False,
    ):
        self.market = market
        self.stream = stream
        self.strategy = strategy
        self.exe = execution or ExecutionConfig()
        self.record_trades = record_trades

    # ------------------------------------------------------------------
    def run(self) -> SimResult:  # noqa: C901 - a single hot loop, kept flat for speed
        mkt, exe, strat = self.market, self.exe, self.strategy
        stream = self.stream
        t_start, t_end = mkt.warmup, stream.horizon
        lat, qsize, maxinv = exe.latency, exe.quote_size, exe.max_inventory
        maker_fee = exe.maker_fee

        times = stream.times.tolist()
        types = stream.types.tolist()
        dist = stream.distance.tolist()
        size = stream.size.tolist()
        life = stream.lifetime.tolist()
        n_ev = len(times)
        news_t = stream.news_times.tolist()
        news_j = stream.news_jumps.tolist()
        news_d = stream.news_delays.tolist()
        n_news = len(news_t)
        sweeps: list[tuple[float, int, float]] = []  # (time, jump, new fair value)
        protect: tuple[int, float, float] | None = None  # (jump, fair value, valid until)

        book = OrderBook()
        orders = book.orders
        best = book.best
        submit_limit, submit_market, cancel = book.submit_limit, book.submit_market, book.cancel
        heap: list[tuple[float, int]] = []
        push, pop = heapq.heappush, heapq.heappop
        for k, (side, price, qty, lt) in enumerate(stream.init_orders):
            oid = n_ev + k
            submit_limit(oid, side, price, qty, BG)
            push(heap, (lt, oid))

        actions: deque = deque()
        my: tuple[dict[int, _MyOrder], dict[int, _MyOrder]] = ({}, {})
        next_my_id = -1
        counters = {"orders": 0, "cancels": 0, "rejects": 0, "late": 0, "fill_cxl": 0, "sniped": 0, "news": 0}

        cash = 0.0
        inv = 0
        spread_capture = inv_pnl = fees = 0.0
        fill_t: list[float] = []
        fill_side: list[int] = []
        fill_px: list[int] = []
        fill_q: list[int] = []
        fill_mid: list[float] = []
        tr_t: list[float] = []
        tr_s: list[int] = []

        mid = book.mid()
        mid_t, mid_v = [0.0], [mid]

        g_t: list[float] = []
        g_mid: list[float] = []
        g_spr: list[int] = []
        g_qb: list[int] = []
        g_qa: list[int] = []
        g_cash: list[float] = []
        g_inv: list[int] = []
        g_quoting: list[bool] = []
        next_grid = t_start
        dt_grid = exe.sample_dt

        if strat is not None:
            strat.reset(SimContext(t_start, t_end, lat, qsize, maxinv, mkt.tick_size))

        def reconcile(side: int, px: int | None, t: float) -> None:
            nonlocal next_my_id
            book_side = my[side]
            keep = False
            for oid, mo in book_side.items():
                if mo.state == _PCXL:
                    continue
                if px is not None and not keep and mo.price == px:
                    keep = True
                    continue
                mo.state = _PCXL
                actions.append((t + lat, _CXL, side, oid))
                counters["cancels"] += 1
            if px is not None and not keep:
                oid = next_my_id
                next_my_id -= 1
                book_side[oid] = _MyOrder(px, qsize, _PNEW)
                actions.append((t + lat, _NEW, side, oid))
                counters["orders"] += 1

        f_bb: list[int] = []   # market (ex-own) best bid / ask and sizes just before each of our fills
        f_ba: list[int] = []
        f_qb: list[int] = []
        f_qa: list[int] = []
        f_sn: list[bool] = []

        def context(aggressor: int):
            """Pre-trade market state, captured only when one of our orders could be hit."""
            if not my[1 - aggressor]:
                return None
            vb, qb = book.best_excluding(BID, MM)
            va, qa = book.best_excluding(ASK, MM)
            return (-1 if vb is None else vb, -1 if va is None else va, qb, qa)

        def take_fills(fills, aggressor: int, m0: float, t: float, sniped: bool, ctx) -> None:
            nonlocal inv, cash, spread_capture, fees
            for oid, owner, px, q in fills:
                if owner != MM:
                    continue
                f_bb.append(ctx[0])
                f_ba.append(ctx[1])
                f_qb.append(ctx[2])
                f_qa.append(ctx[3])
                f_sn.append(sniped)
                maker_side = 1 - aggressor
                mo = my[maker_side][oid]
                if maker_side == BID:
                    inv += q
                    cash -= px * q
                    spread_capture += (m0 - px) * q
                    fill_side.append(1)
                else:
                    inv -= q
                    cash += px * q
                    spread_capture += (px - m0) * q
                    fill_side.append(-1)
                cash -= maker_fee * q
                fees += maker_fee * q
                fill_t.append(t)
                fill_px.append(px)
                fill_q.append(q)
                fill_mid.append(m0)
                if mo.state == _PCXL:
                    counters["fill_cxl"] += 1
                if sniped:
                    counters["sniped"] += q
                mo.qty -= q
                if mo.qty == 0:
                    del my[maker_side][oid]

        INF = math.inf
        i = jn = 0
        while True:
            t, kind = (actions[0][0], _ACTION) if actions else (INF, _ACTION)
            if jn < n_news and news_t[jn] < t:
                t, kind = news_t[jn], _NEWS
            if sweeps and sweeps[0][0] < t:
                t, kind = sweeps[0][0], _SWEEP
            if heap and heap[0][0] < t:
                t, kind = heap[0][0], _CANCEL
            if i < n_ev and times[i] < t:
                t, kind = times[i], _EVENT
            if t >= t_end:
                break

            while t >= next_grid:  # sample the (piecewise-constant) state on a regular grid
                g_t.append(next_grid)
                g_mid.append(mid)
                bb, ba = best
                g_spr.append(ba - bb if bb is not None and ba is not None else -1)
                g_qb.append(book.best_qty(BID))
                g_qa.append(book.best_qty(ASK))
                g_cash.append(cash)
                g_inv.append(inv)
                g_quoting.append(any(o in orders for o in my[BID]) and any(o in orders for o in my[ASK]))
                next_grid += dt_grid

            if kind == _EVENT:
                et = types[i]
                if et <= MS:
                    aggressor = BUY if et == MB else SELL
                    ctx = context(aggressor)
                    take_fills(submit_market(aggressor, size[i]), aggressor, mid, t, False, ctx)
                    if self.record_trades:
                        tr_t.append(t)
                        tr_s.append(1 if et == MB else -1)
                else:
                    side = BID if et == LB else ASK
                    opp = best[1 - side]
                    if opp is None:  # one-sided book: behave as if the spread were one tick
                        own = best[side]
                        if own is None:
                            own = int(math.floor(mid_v[-1])) if side == BID else int(math.ceil(mid_v[-1]))
                        opp = own + 1 if side == BID else own - 1
                    price = opp - dist[i] if side == BID else opp + dist[i]
                    submit_limit(i, side, price, size[i], BG)
                    push(heap, (t + life[i], i))
                if strat is not None:
                    strat.on_market_event(t, et)
                i += 1
            elif kind == _CANCEL:
                _, oid = pop(heap)
                if oid not in orders:
                    continue  # already executed
                cancel(oid)
            elif kind == _NEWS:
                jump, delay = news_j[jn], news_d[jn]
                jn += 1
                fair = (mid if mid is not None else mid_v[-1]) + jump
                heapq.heappush(sweeps, (t + delay, jump, fair))
                if exe.news_protection:
                    protect = (jump, fair, t + exe.news_window)
                counters["news"] += t >= t_start
            elif kind == _SWEEP:
                _, jump, fair = heapq.heappop(sweeps)
                if jump > 0:   # take every ask priced strictly below the new fair value
                    ctx = context(BUY)
                    take_fills(book.submit_ioc(BUY, 1 << 40, math.ceil(fair) - 1), BUY, mid, t, True, ctx)
                else:          # hit every bid priced strictly above it
                    ctx = context(SELL)
                    take_fills(book.submit_ioc(SELL, 1 << 40, math.floor(fair) + 1), SELL, mid, t, True, ctx)
            else:
                _, akind, side, oid = actions.popleft()
                mo = my[side].get(oid)
                if akind == _NEW:
                    if mo is None:
                        continue
                    if book.would_cross(side, mo.price):  # post-only: reject rather than take
                        del my[side][oid]
                        counters["rejects"] += 1
                    else:
                        submit_limit(oid, side, mo.price, mo.qty, MM)
                        if mo.state == _PNEW:
                            mo.state = _LIVE
                else:
                    if mo is None:
                        counters["late"] += 1  # the order traded (or was rejected) first
                        continue
                    cancel(oid)
                    del my[side][oid]

            new_mid = book.mid()
            if new_mid is not None and new_mid != mid:
                if mid is not None:
                    inv_pnl += inv * (new_mid - mid)
                mid = new_mid
                mid_t.append(t)
                mid_v.append(mid)

            if strat is not None and t >= t_start and kind != _ACTION:
                vb, qb = book.best_excluding(BID, MM)
                va, qa = book.best_excluding(ASK, MM)
                if vb is None or va is None:
                    bid_px = ask_px = None
                else:
                    my_b = my_a = None
                    for mo in my[BID].values():
                        if mo.state != _PCXL:
                            my_b = mo.price
                            break
                    for mo in my[ASK].values():
                        if mo.state != _PCXL:
                            my_a = mo.price
                            break
                    bid_px, ask_px = strat.quote(t, MarketView(book, vb, va, qb, qa, my_b, my_a), inv)
                    if protect is not None and t < protect[2]:  # stay out of the arbitrageur's way
                        jump, fair = protect[0], protect[1]
                        if jump > 0 and ask_px is not None and ask_px < fair:
                            ask_px = math.ceil(fair)
                        elif jump < 0 and bid_px is not None and bid_px > fair:
                            bid_px = math.floor(fair)
                    bb, ba = best
                    if bid_px is not None and ba is not None and bid_px >= ba:
                        bid_px = ba - 1
                    if ask_px is not None and bb is not None and ask_px <= bb:
                        ask_px = bb + 1
                if inv >= maxinv:
                    bid_px = None
                if inv <= -maxinv:
                    ask_px = None
                reconcile(BID, bid_px, t)
                reconcile(ASK, ask_px, t)

        self.book = book  # kept for inspection (tests, debugging)

        # -------- close out: sweep the remaining inventory through the final book (read-only)
        liquidation = 0.0
        final_mid = float(mid)
        if inv != 0:
            side = SELL if inv > 0 else BUY
            notional, filled = book.sweep_cost(side, abs(inv))
            worst = (best[BID] if side == SELL else best[ASK]) or final_mid
            unfilled = abs(inv) - filled  # if the book is too thin, assume a punitive 5-tick slippage
            notional += unfilled * (worst - 5 if side == SELL else worst + 5)
            if side == SELL:
                liquidation = inv * final_mid - notional
            else:
                liquidation = notional - (-inv) * final_mid
            liquidation += exe.taker_fee * abs(inv)

        f = np.asarray
        return SimResult(
            seed=stream.seed,
            strategy=strat.name if strat is not None else "none",
            t_start=t_start,
            t_end=t_end,
            mid_t=f(mid_t, float),
            mid_v=f(mid_v, float),
            grid_t=f(g_t, float),
            grid_mid=f(g_mid, float),
            grid_spread=f(g_spr, np.int64),
            grid_qbid=f(g_qb, np.int64),
            grid_qask=f(g_qa, np.int64),
            grid_cash=f(g_cash, float),
            grid_inv=f(g_inv, np.int64),
            grid_quoting=f(g_quoting, bool),
            fill_t=f(fill_t, float),
            fill_side=f(fill_side, np.int64),
            fill_price=f(fill_px, np.int64),
            fill_qty=f(fill_q, np.int64),
            fill_mid=f(fill_mid, float),
            spread_capture=spread_capture,
            inventory_pnl=inv_pnl,
            fees=fees,
            liquidation_cost=liquidation,
            final_cash=cash,
            final_inventory=inv,
            final_mid=final_mid,
            n_events=n_ev,
            n_orders=counters["orders"],
            n_cancels=counters["cancels"],
            n_rejects=counters["rejects"],
            n_late_cancels=counters["late"],
            n_fill_while_cancelling=counters["fill_cxl"],
            trades_t=f(tr_t, float),
            trades_sign=f(tr_s, np.int64),
            n_news=counters["news"],
            sniped_qty=counters["sniped"],
            fill_best_bid=f(f_bb, np.int64),
            fill_best_ask=f(f_ba, np.int64),
            fill_qty_bid=f(f_qb, np.int64),
            fill_qty_ask=f(f_qa, np.int64),
            fill_sniped=f(f_sn, bool),
        )


def run_episode(
    market: MarketConfig,
    seed: int,
    duration: float,
    strategy: Strategy | None = None,
    execution: ExecutionConfig | None = None,
    record_trades: bool = False,
) -> SimResult:
    """Convenience wrapper: draw the stream for ``seed`` and simulate warm-up + ``duration``."""
    from .market import generate_stream

    stream = generate_stream(market, market.warmup + duration, seed)
    return Simulator(market, stream, strategy, execution, record_trades).run()
