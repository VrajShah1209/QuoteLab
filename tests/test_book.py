import numpy as np
import pytest

from quotelab.book import ASK, BID, BUY, SELL, OrderBook


class ReferenceBook:
    """Deliberately naive O(n) book: a flat list scanned and sorted on every operation."""

    def __init__(self):
        self.orders = []  # dicts: oid, side, price, qty, owner, seq
        self.seq = 0

    def _crosses(self, side, price):
        opp = [o["price"] for o in self.orders if o["side"] != side]
        if not opp:
            return False
        return price >= min(opp) if side == BID else price <= max(opp)

    def _match(self, side, qty, limit):
        fills = []
        while qty > 0:
            opp = [o for o in self.orders if o["side"] != side]
            if limit is not None:
                opp = [o for o in opp if (o["price"] <= limit if side == BUY else o["price"] >= limit)]
            if not opp:
                break
            key = (lambda o: (o["price"], o["seq"])) if side == BUY else (lambda o: (-o["price"], o["seq"]))
            o = min(opp, key=key)
            take = min(qty, o["qty"])
            o["qty"] -= take
            qty -= take
            fills.append((o["oid"], o["owner"], o["price"], take))
            if o["qty"] == 0:
                self.orders.remove(o)
        return fills, qty

    def limit(self, oid, side, price, qty, owner, post_only):
        fills = []
        if self._crosses(side, price):
            if post_only:
                return "rejected", []
            fills, qty = self._match(side, qty, price)
            if qty == 0:
                return "filled", fills
        self.seq += 1
        self.orders.append(dict(oid=oid, side=side, price=price, qty=qty, owner=owner, seq=self.seq))
        return ("partial" if fills else "resting"), fills

    def market(self, side, qty):
        return self._match(side, qty, None)[0]

    def cancel(self, oid):
        for o in self.orders:
            if o["oid"] == oid:
                self.orders.remove(o)
                return o["qty"]
        return 0

    def snapshot(self):
        out = []
        for side in (BID, ASK):
            prices = sorted({o["price"] for o in self.orders if o["side"] == side}, reverse=(side == BID))
            out.append([
                (p, [(o["oid"], o["qty"], o["owner"]) for o in sorted(
                    (o for o in self.orders if o["side"] == side and o["price"] == p), key=lambda o: o["seq"])])
                for p in prices
            ])
        return out[0], out[1]


def test_fifo_priority_at_one_price():
    b = OrderBook()
    b.submit_limit(1, ASK, 101, 2, owner=0)
    b.submit_limit(2, ASK, 101, 3, owner=1)
    b.submit_limit(3, ASK, 101, 1, owner=0)
    fills = b.submit_market(BUY, 4)
    assert fills == [(1, 0, 101, 2), (2, 1, 101, 2)]
    assert b.queue_ahead(2) == 0 and b.queue_ahead(3) == 1
    assert b.level_qty(ASK, 101) == 2
    b.check_invariants()


def test_price_priority_and_walking_the_book():
    b = OrderBook()
    for oid, px in [(1, 103), (2, 101), (3, 102)]:
        b.submit_limit(oid, ASK, px, 1)
    b.submit_limit(4, BID, 99, 5)
    fills = b.submit_market(BUY, 3)
    assert [f[2] for f in fills] == [101, 102, 103]
    assert b.best_ask is None and b.best_bid == 99 and b.mid() is None
    assert b.submit_market(BUY, 2) == []  # no liquidity: nothing happens
    b.check_invariants()


def test_cancel_updates_best_and_depth():
    b = OrderBook()
    b.submit_limit(1, BID, 100, 2)
    b.submit_limit(2, BID, 99, 3)
    b.submit_limit(3, ASK, 102, 1)
    assert b.spread() == 2 and b.mid() == 101.0
    assert b.cancel(1) == 2
    assert b.best_bid == 99
    assert b.cancel(1) == 0  # cancelling twice is a no-op ("too late")
    assert b.depth(BID, 5) == [(99, 3)]
    b.check_invariants()


def test_post_only_and_marketable_limit():
    b = OrderBook()
    b.submit_limit(1, ASK, 101, 2)
    assert b.submit_limit(2, BID, 101, 1, post_only=True) == ("rejected", [])
    status, fills = b.submit_limit(3, BID, 102, 5)
    assert status == "partial" and fills == [(1, 0, 101, 2)]
    assert b.best_bid == 102 and b.level_qty(BID, 102) == 3 and b.best_ask is None
    b.check_invariants()


def test_sweep_cost_is_read_only():
    b = OrderBook()
    b.submit_limit(1, BID, 100, 2)
    b.submit_limit(2, BID, 98, 5)
    before = b.snapshot()
    assert b.sweep_cost(SELL, 4) == (2 * 100 + 2 * 98, 4)
    assert b.sweep_cost(SELL, 50) == (2 * 100 + 5 * 98, 7)
    assert b.snapshot() == before


def test_duplicate_ids_rejected():
    b = OrderBook()
    b.submit_limit(1, BID, 100, 1)
    with pytest.raises(ValueError):
        b.submit_limit(1, BID, 99, 1)


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_against_reference_implementation(seed):
    """Random operation streams must produce identical fills and identical L3 state."""
    rng = np.random.default_rng(seed)
    fast, ref = OrderBook(), ReferenceBook()
    live: list[int] = []
    next_id = 0
    for _ in range(1500):
        u = rng.random()
        if u < 0.55:  # limit order, sometimes marketable
            side = int(rng.integers(2))
            price = int(rng.integers(90, 111))
            qty = int(rng.integers(1, 6))
            owner = int(rng.integers(2))
            post_only = bool(rng.random() < 0.3)
            next_id += 1
            r1 = fast.submit_limit(next_id, side, price, qty, owner, post_only)
            r2 = ref.limit(next_id, side, price, qty, owner, post_only)
            assert r1 == r2
            live.append(next_id)
        elif u < 0.75:  # market order
            side = int(rng.integers(2))
            qty = int(rng.integers(1, 12))
            assert fast.submit_market(side, qty) == ref.market(side, qty)
        elif live:  # cancel a random previously-submitted order (may already be gone)
            oid = live.pop(int(rng.integers(len(live))))
            assert fast.cancel(oid) == ref.cancel(oid)
        assert fast.snapshot() == ref.snapshot()
        fast.check_invariants()
