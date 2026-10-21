"""Command line: ``python -m quotelab demo`` or ``quotelab simulate --strategy mk --gamma 0.1``.

When run from the repository root after the experiments, the commands use the calibrated
parameters in ``results/calibration.json`` and the validated configurations in
``results/validation.json``; otherwise they fall back to built-in defaults.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .analytics import episode_metrics
from .experiment import build_strategy
from .market import MarketConfig, generate_stream
from .simulator import ExecutionConfig, Simulator

# Fallbacks (ticks, seconds), equal to the calibration shipped in results/calibration.json.
FALLBACK = {"sigma": 0.778, "A": 1.375, "k": 1.845}
NAMES = {"join": "Join BBO", "as": "Avellaneda-Stoikov", "glft": "GLFT", "mk": "MK-full (markout GLFT)"}


def _load(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def specs_for(results_dir: Path, gamma: float | None, queue_premium: float | None) -> tuple[dict, str]:
    """Strategy specs, from validated results if available."""
    cal = _load(results_dir / "calibration.json")
    val = _load(results_dir / "validation.json")
    c = cal or FALLBACK
    kappa = queue_premium if queue_premium is not None else (val["queue_premium"] if val else 0.5)
    specs = {
        "join": {"kind": "join"},
        "as": {"kind": "as", "gamma": 3e-4, "sigma": c["sigma"], "k": c["k"], "queue_premium": kappa},
        "glft": {"kind": "glft", "gamma": 0.1, "sigma": c["sigma"], "A": c["A"], "k": c["k"], "queue_premium": kappa},
    }
    source = "built-in defaults"
    if val:
        sel = val["selected"]
        specs["join"] = dict(sel["join_bbo"])
        specs["as"] = dict(sel["avellaneda_stoikov"])
        specs["glft"] = dict(sel["glft"])
        specs["mk"] = dict(sel["mk_full"])
        source = f"{results_dir}/validation.json"
    elif cal:
        specs["mk"] = {"kind": "glft_markout", "gamma": 0.1, "sigma": c["sigma"], "A": c["A"], "k": c["k"],
                       "queue_premium": kappa, "markout": cal["markout"]["full"]["model"]}
        source = f"{results_dir}/calibration.json"
    for name, s in specs.items():
        s["label"] = NAMES[name]
        if gamma is not None and "gamma" in s:
            s["gamma"] = gamma
        if queue_premium is not None and "queue_premium" in s:
            s["queue_premium"] = queue_premium
    return specs, source


HEADER = f"{'strategy':24s} {'PnL':>8s} {'fills':>6s} {'edge':>6s} {'mkout1s':>8s} {'|q|':>5s} {'msgs/fill':>9s}"


def _row(m: dict) -> str:
    return (f"{m['label']:24s} {m['pnl']:8.1f} {m['n_fills']:6d} {m['edge_per_unit']:6.3f} "
            f"{m['markout_1s']:8.3f} {m['mean_abs_inventory']:5.2f} {m['order_to_trade']:9.2f}")


def _run(spec: dict, stream, cfg: MarketConfig, exe: ExecutionConfig) -> dict:
    res = Simulator(cfg, stream, build_strategy(spec), exe).run()
    m = episode_metrics(res)
    m["label"] = spec["label"]
    return m


def cmd_demo(args: argparse.Namespace) -> None:
    specs, source = specs_for(Path(args.results), None, None)
    cfg = MarketConfig()
    stream = generate_stream(cfg, cfg.warmup + 60 * args.minutes, args.seed)
    exe = ExecutionConfig(latency=args.latency_ms / 1000)
    print(f"seed {args.seed}: {len(stream):,} order-flow events and {stream.news_times.size} news items over "
          f"{args.minutes:g} min (+{cfg.warmup:.0f} s warm-up), latency {args.latency_ms:g} ms")
    print(f"parameters: {source}. PnL in ticks x lots; edge and markout in ticks per lot.\n")
    print(HEADER)
    for name in ("join", "as", "glft", "mk"):
        if name in specs:
            t0 = time.perf_counter()
            m = _run(specs[name], stream, cfg, exe)
            print(_row(m) + f"   ({time.perf_counter() - t0:.1f}s)")


def cmd_simulate(args: argparse.Namespace) -> None:
    specs, _ = specs_for(Path(args.results), args.gamma, args.queue_premium)
    if args.strategy not in specs:
        raise SystemExit("the markout strategy needs results/calibration.json (run experiments/02_calibrate.py)")
    cfg = MarketConfig()
    stream = generate_stream(cfg, cfg.warmup + 60 * args.minutes, args.seed)
    exe = ExecutionConfig(latency=args.latency_ms / 1000, max_inventory=args.max_inventory)
    m = _run(specs[args.strategy], stream, cfg, exe)
    if args.json:
        print(json.dumps(m, indent=2))
    else:
        print(HEADER)
        print(_row(m))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quotelab", description="Hawkes-driven LOB simulation and market making.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=7)
    common.add_argument("--minutes", type=float, default=30.0)
    common.add_argument("--latency-ms", type=float, default=5.0)
    common.add_argument("--results", default="results", help="directory with calibration/validation JSON")
    d = sub.add_parser("demo", parents=[common], help="run the strategies on one episode")
    d.set_defaults(func=cmd_demo)
    s = sub.add_parser("simulate", parents=[common], help="run one strategy")
    s.add_argument("--strategy", choices=["join", "as", "glft", "mk"], default="mk")
    s.add_argument("--gamma", type=float, default=None)
    s.add_argument("--queue-premium", type=float, default=None)
    s.add_argument("--max-inventory", type=int, default=10)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_simulate)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
