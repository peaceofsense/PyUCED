"""
DE-LU MILP Day-Ahead Unit Commitment & Economic Dispatch
Germany (DE-LU) | 10 May 2026 | 96 × 15-min intervals

Pipeline:
  1. data_processing  — clean raw ENTSO-E CSVs → data/processed/
  2. model            — build & solve MILP      → results/
  3. plot             — render figures           → plots/

Usage:
    python main.py

Or run each stage independently:
    python src/data_processing.py
    python src/model.py
    python src/plot.py

Dependencies:
    pip install pyomo highspy pandas matplotlib numpy
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processing import load_and_process
from model import build_and_solve
from plot import make_plots


def main():
    print("=" * 60)
    print("DE-LU MILP Unit Commitment & Economic Dispatch")
    print("Germany (DE-LU) | 10 May 2026 | 96 × 15-min intervals")
    print("=" * 60)

    print("\n[1/3] Data Processing")
    print("-" * 40)
    load_and_process()

    print("\n[2/3] MILP Model — Build & Solve")
    print("-" * 40)
    build_and_solve()

    print("\n[3/3] Generating Plots")
    print("-" * 40)
    make_plots()

    print("\n" + "=" * 60)
    print("Done. Outputs written to:")
    print("  data/processed/   — demand.csv, generation_pivot.csv, residual.csv")
    print("  results/          — dispatch_extended.csv, commitment_extended.csv,")
    print("                      gen_costs.csv, co2_tonnes.csv, summary.json")
    print("  plots/            — dispatch_extended.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
