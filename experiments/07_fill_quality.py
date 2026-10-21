"""Experiment 7: fill quality by queue position (first 100 test episodes, configurations frozen).

Every fill is classified by where the order sat relative to the market's best price on its side just
before the trade (own orders excluded): improving the spread, at the touch, or behind it. Fills taken
by the news arbitrageur form a fourth class. For each class: share of volume, edge at the fill,
5-second markout, realised spread (edge + markout), and the realised spread summed per episode.

Writes results/fill_quality.json.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from _common import DURATION, LATENCY, TEST, WORKERS, load_json, save_json, step

from quotelab.analytics import mid_at
from quotelab.experiment import build_strategy
from quotelab.market import MarketConfig, generate_stream
from quotelab.simulator import ExecutionConfig, Simulator

SEEDS = TEST[:100]
FAMILIES = ["mk_full", "glft", "join_bbo"]
CLASSES = ["improving", "at_touch", "behind", "sniped"]
HORIZON = 5.0


def classify(res) -> np.ndarray:
    """0 improving, 1 at the touch, 2 behind, 3 taken by the news arbitrageur."""
    ref = np.where(res.fill_side > 0, res.fill_best_bid, res.fill_best_ask)
    behind = np.where(res.fill_side > 0, ref - res.fill_price, res.fill_price - ref)
    cls = np.where(behind < 0, 0, np.where(behind == 0, 1, 2))
    cls = np.where(ref < 0, 0, cls)  # no other order on our side: we were the best price
    return np.where(res.fill_sniped, 3, cls)


def _worker(args: tuple) -> dict:
    seed, specs = args
    cfg = MarketConfig()
    stream = generate_stream(cfg, cfg.warmup + DURATION, seed)
    out = {}
    for fam, spec in specs.items():
        res = Simulator(cfg, stream, build_strategy(spec), ExecutionConfig(latency=LATENCY)).run()
        keep = res.fill_t + HORIZON <= res.t_end
        q = res.fill_qty[keep].astype(float)
        cls = classify(res)[keep]
        edge = (res.fill_side * (res.fill_mid - res.fill_price))[keep]
        mk = (res.fill_side * (mid_at(res, res.fill_t + HORIZON) - res.fill_mid))[keep]
        out[fam] = {c: [float(q[cls == i].sum()), float((q * edge)[cls == i].sum()),
                        float((q * mk)[cls == i].sum())] for i, c in enumerate(CLASSES)}
    return out


def main() -> None:
    val = load_json("validation.json")
    specs = {fam: val["selected"][fam] for fam in FAMILIES}
    with step(f"fill quality: {len(FAMILIES)} strategies x {len(SEEDS)} test episodes"):
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            parts = list(ex.map(_worker, [(s, specs) for s in SEEDS], chunksize=1))
    out: dict = {"seeds": SEEDS, "horizon": HORIZON, "strategies": {}}
    for fam in FAMILIES:
        tot = np.array([[p[fam][c] for c in CLASSES] for p in parts]).sum(axis=0)  # (class, [q, q*edge, q*mk])
        volume = tot[:, 0].sum()
        out["strategies"][fam] = {
            c: {
                "share": float(tot[i, 0] / volume),
                "edge": float(tot[i, 1] / tot[i, 0]) if tot[i, 0] else float("nan"),
                "markout": float(tot[i, 2] / tot[i, 0]) if tot[i, 0] else float("nan"),
                "realised": float((tot[i, 1] + tot[i, 2]) / tot[i, 0]) if tot[i, 0] else float("nan"),
                "per_episode": float((tot[i, 1] + tot[i, 2]) / len(SEEDS)),
                "lots_per_episode": float(tot[i, 0] / len(SEEDS)),
            } for i, c in enumerate(CLASSES)
        }
    save_json("fill_quality.json", out)
    for fam, d in out["strategies"].items():
        print(f"  {fam:10s} " + "  ".join(f"{c}: {v['share']:.3f} rs {v['realised']:+.3f}" for c, v in d.items()))


if __name__ == "__main__":
    main()
