"""Level-3 limit order book with price-time (FIFO) priority.

Prices are integer ticks. Each price level keeps its resting orders in an ``OrderedDict`` so
that the head of the queue is O(1) to match and any order is O(1) to cancel. Only non-empty
levels are stored, and the best bid / ask are maintained incrementally.

Conventions
-----------
* ``side`` of a resting order: ``BID`` (0) or ``ASK`` (1).
* ``side`` of an aggressive (market) order: ``BUY`` (= BID, consumes asks) or ``SELL``.
* A fill is reported as ``(maker_oid, maker_owner, price, qty)``.
"""

from __future__ import annotations

from collections import OrderedDict

BID, ASK = 0, 1
BUY, SELL = BID, ASK

Fill = tuple[int, int, int, int]


class Order:
    __slots__ = ("oid", "side", "price", "qty", "owner")

    def __init__(self, oid: int, side: int, price: int, qty: int, owner: int):
        self.oid, self.side, self.price, self.qty, self.owner = oid, side, price, qty, owner

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        s = "BID" if self.side == BID else "ASK"
        return f"Order({self.oid}, {s}, px={self.price}, qty={self.qty}, owner={self.owner})"


class Level:
    __slots__ = ("price", "qty", "orders")

    def __init__(self, price: int):
        self.price = price
        self.qty = 0
        self.orders: OrderedDict[int, Order] = OrderedDict()


class OrderBook:
    """A single-instrument limit order book."""

    def __init__(self) -> None:
        self._levels: tuple[dict[int, Level], dict[int, Level]] = ({}, {})
        self.best: list[int | None] = [None, None]
        self.orders: dict[int, Order] = {}

    # ------------------------------------------------------------------ queries
    @property
    def best_bid(self) -> int | None:
        return self.best[BID]

    @property
    def best_ask(self) -> int | None:
        return self.best[ASK]

    def mid(self) -> float | None:
        b, a = self.best
        return None if b is None or a is None else 0.5 * (b + a)

    def spread(self) -> int | None:
        b, a = self.best
        return None if b is None or a is None else a - b

    def level_qty(self, side: int, price: int) -> int:
        lv = self._levels[side].get(price)
        return 0 if lv is None else lv.qty

    def best_qty(self, side: int) -> int:
        p = self.best[side]
        return 0 if p is None else self._levels[side][p].qty

    def depth(self, side: int, n_levels: int) -> list[tuple[int, int]]:
        """Aggregated (price, qty) for the best ``n_levels`` non-empty levels."""
        lv = self._levels[side]
        prices = sorted(lv, reverse=(side == BID))[:n_levels]
        return [(p, lv[p].qty) for p in prices]

    def total_qty(self, side: int) -> int:
        return sum(lv.qty for lv in self._levels[side].values())

    def queue_ahead(self, oid: int) -> int:
        """Quantity resting ahead of order ``oid`` at its price level."""
        o = self.orders[oid]
        ahead = 0
        for other_id, other in self._levels[o.side][o.price].orders.items():
            if other_id == oid:
                return ahead
            ahead += other.qty
        raise KeyError(oid)  # pragma: no cover - inconsistent book

    def best_excluding(self, side: int, owner: int) -> tuple[int | None, int]:
        """(price, qty) of the best level on ``side`` after removing ``owner``'s orders.

        A market maker computes its fair value from the market's book, not from a book that
        contains its own quotes; otherwise its quotes feed back into its own signal.
        """
        lv = self._levels[side]
        p = self.best[side]
        while p is not None:
            level = lv[p]
            q = level.qty
            for o in level.orders.values():
                if o.owner == owner:
                    q -= o.qty
            if q > 0:
                return p, q
            p = self._next_best(side, p)
        return None, 0

    def sweep_cost(self, side: int, qty: int) -> tuple[int, int]:
        """Read-only: (notional in ticks, filled qty) of a market order of ``qty`` on ``side``."""
        opp = 1 - side
        lv = self._levels[opp]
        notional = filled = 0
        for p in sorted(lv, reverse=(opp == BID)):
            take = min(qty - filled, lv[p].qty)
            notional += take * p
            filled += take
            if filled == qty:
                break
        return notional, filled

    def snapshot(self) -> tuple[list, list]:
        """Full L3 state, best levels first: [(price, [(oid, qty, owner), ...]), ...] per side."""
        out = []
        for side in (BID, ASK):
            lv = self._levels[side]
            out.append([
                (p, [(o.oid, o.qty, o.owner) for o in lv[p].orders.values()])
                for p in sorted(lv, reverse=(side == BID))
            ])
        return out[0], out[1]

    # ---------------------------------------------------------------- mutations
    def _next_best(self, side: int, from_price: int) -> int | None:
        lv = self._levels[side]
        if not lv:
            return None
        step = -1 if side == BID else 1
        p = from_price + step
        for _ in range(64):  # the book is dense near the top: almost always found here
            if p in lv:
                return p
            p += step
        if side == BID:
            rest = [x for x in lv if x < from_price]
            return max(rest) if rest else None
        rest = [x for x in lv if x > from_price]
        return min(rest) if rest else None

    def _rest(self, oid: int, side: int, price: int, qty: int, owner: int) -> None:
        if oid in self.orders:
            raise ValueError(f"duplicate order id {oid}")
        lv = self._levels[side].get(price)
        if lv is None:
            lv = Level(price)
            self._levels[side][price] = lv
        o = Order(oid, side, price, qty, owner)
        lv.orders[oid] = o
        lv.qty += qty
        self.orders[oid] = o
        best = self.best[side]
        if best is None or (price > best if side == BID else price < best):
            self.best[side] = price

    def _match(self, side: int, qty: int, limit: int | None) -> tuple[list[Fill], int]:
        """Match an aggressor on ``side`` against the opposite book, best price first, FIFO."""
        opp = 1 - side
        levels = self._levels[opp]
        fills: list[Fill] = []
        while qty > 0:
            price = self.best[opp]
            if price is None or (limit is not None and (price > limit if side == BUY else price < limit)):
                break
            lv = levels[price]
            queue = lv.orders
            while qty > 0 and queue:
                oid, o = next(iter(queue.items()))
                take = o.qty if o.qty < qty else qty
                o.qty -= take
                lv.qty -= take
                qty -= take
                fills.append((oid, o.owner, price, take))
                if o.qty == 0:
                    queue.popitem(last=False)
                    del self.orders[oid]
            if not queue:
                del levels[price]
                self.best[opp] = self._next_best(opp, price)
        return fills, qty

    def submit_market(self, side: int, qty: int) -> list[Fill]:
        """Aggressive order for ``qty``; any quantity beyond available liquidity is dropped."""
        return self._match(side, qty, None)[0]

    def submit_ioc(self, side: int, qty: int, limit: int) -> list[Fill]:
        """Immediate-or-cancel: take liquidity up to ``limit`` (inclusive); nothing rests."""
        return self._match(side, qty, limit)[0]

    def would_cross(self, side: int, price: int) -> bool:
        opp = self.best[1 - side]
        return opp is not None and (price >= opp if side == BID else price <= opp)

    def submit_limit(
        self, oid: int, side: int, price: int, qty: int, owner: int = 0, post_only: bool = False
    ) -> tuple[str, list[Fill]]:
        """Limit order. Returns (status, fills); status in {'resting', 'filled', 'partial', 'rejected'}.

        A marketable order executes against the opposite side up to ``price`` and the remainder
        rests. With ``post_only`` a marketable order is rejected instead (maker-only semantics).
        """
        if qty <= 0:
            raise ValueError("quantity must be positive")
        if self.would_cross(side, price):
            if post_only:
                return "rejected", []
            fills, rem = self._match(side, qty, price)
            if rem == 0:
                return "filled", fills
            self._rest(oid, side, price, rem, owner)
            return "partial", fills
        self._rest(oid, side, price, qty, owner)
        return "resting", []

    def cancel(self, oid: int) -> int:
        """Cancel a resting order. Returns the cancelled quantity (0 if it no longer rests)."""
        o = self.orders.pop(oid, None)
        if o is None:
            return 0
        levels = self._levels[o.side]
        lv = levels[o.price]
        del lv.orders[oid]
        lv.qty -= o.qty
        if not lv.orders:
            del levels[o.price]
            if self.best[o.side] == o.price:
                self.best[o.side] = self._next_best(o.side, o.price)
        return o.qty

    def check_invariants(self) -> None:
        """Raise AssertionError if the book is internally inconsistent (used by tests)."""
        for side in (BID, ASK):
            lv = self._levels[side]
            expected_best = (max(lv) if side == BID else min(lv)) if lv else None
            assert self.best[side] == expected_best, (side, self.best[side], expected_best)
            for p, level in lv.items():
                assert level.orders, "empty level stored"
                assert level.qty == sum(o.qty for o in level.orders.values())
                for o in level.orders.values():
                    assert o.qty > 0 and o.price == p and o.side == side
                    assert self.orders[o.oid] is o
        n = sum(len(level.orders) for side in (BID, ASK) for level in self._levels[side].values())
        assert n == len(self.orders)
        b, a = self.best
        assert b is None or a is None or b < a, "crossed book"
