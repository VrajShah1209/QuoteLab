"""Reproduce every result and figure: python experiments/run_all.py  (QUOTELAB_FAST=1 for a smoke run)."""

import runpy
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_market_and_hawkes.py", "02_calibrate.py", "03_validate.py", "04_test.py", "05_robustness.py",
         "06_ladder.py", "07_fill_quality.py", "08_tables.py", "09_summary_figure.py"]

if __name__ == "__main__":
    t0 = time.perf_counter()
    for name in STEPS:
        print(f"\n=== {name} ===", flush=True)
        runpy.run_path(str(HERE / name), run_name="__main__")
    print(f"\nall experiments finished in {(time.perf_counter() - t0) / 60:.1f} min")
