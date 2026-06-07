"""
Data loading and preprocessing for the DE-LU MILP Day-Ahead UC/ED model.

Reads raw ENTSO-E CSVs from data/raw/, cleans them, and writes three
processed files to data/processed/:
    demand.csv          — 96-interval day-ahead load (MW)
    generation_pivot.csv — generation by type per interval (MW)
    residual.csv        — demand minus must-run generation (MW)
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

LOAD_CSV = os.path.join(RAW_DIR, "GUI_TOTAL_LOAD_DAYAHEAD_202605092200-202605102200.csv")
GEN_CSV = os.path.join(
    RAW_DIR, "AGGREGATED_GENERATION_PER_TYPE_GENERATION_202605092200-202605102200.csv"
)

TYPE_MAP = {
    "Biomass": "Biomass",
    "Fossil Brown coal/Lignite": "Lignite",
    "Fossil Coal-derived gas": "CoalGas",
    "Fossil Gas": "Gas_CCGT",
    "Fossil Hard coal": "HardCoal",
    "Fossil Oil": "Oil",
    "Fossil Oil shale": "Oil",
    "Fossil Peat": "Other",
    "Geothermal": "Geothermal",
    "Hydro Pumped Storage": "HydroPump",
    "Hydro Run-of-river and pondage": "HydroRoR",
    "Hydro Water Reservoir": "HydroRes",
    "Marine": "Other",
    "Nuclear": "Nuclear",
    "Other": "Other",
    "Other renewable": "OtherRen",
    "Solar": "Solar",
    "Waste": "Waste",
    "Wind Offshore": "WindOff",
    "Wind Onshore": "WindOn",
    "Energy storage": "HydroPump",
}

ALL_TYPES = [
    "WindOn", "WindOff", "Solar", "HydroRoR", "Geothermal",
    "OtherRen", "Other", "Nuclear", "Biomass", "Waste",
    "HydroPump", "HydroRes", "Lignite", "HardCoal", "CoalGas", "Gas_CCGT", "Oil",
]

MUST_RUN = ["WindOn", "WindOff", "Solar", "HydroRoR", "Geothermal", "OtherRen", "Nuclear", "Other"]


def _strip_quotes(df):
    return df.apply(
        lambda c: c.map(lambda x: x.strip('"').strip() if isinstance(x, str) else x)
    )


def _parse_mtu(s):
    s = s.strip('"').strip()
    return pd.to_datetime(s.split(" - ")[0].strip(), dayfirst=True)


def load_and_process():
    """Load raw CSVs, clean, compute residual, save to data/processed/."""
    print("Loading data...")

    for path in (LOAD_CSV, GEN_CSV):
        if not os.path.exists(path):
            print(f"\n  ERROR: File not found:\n    {path}")
            print("  Place both ENTSO-E CSVs in data/raw/")
            sys.exit(1)

    load_df = pd.read_csv(LOAD_CSV, sep=",", quotechar='"', skipinitialspace=True)
    gen_df = pd.read_csv(GEN_CSV, sep=",", quotechar='"', skipinitialspace=True)
    load_df = _strip_quotes(load_df)
    gen_df = _strip_quotes(gen_df)
    load_df.columns = load_df.columns.str.strip().str.strip('"')
    gen_df.columns = gen_df.columns.str.strip().str.strip('"')

    load_df["time"] = load_df["MTU (CET/CEST)"].apply(_parse_mtu)
    gen_df["time"] = gen_df["MTU (CET/CEST)"].apply(_parse_mtu)

    load_df["demand_MW"] = pd.to_numeric(load_df["Actual Total Load (MW)"], errors="coerce")
    demand = load_df.set_index("time")["demand_MW"].sort_index().iloc[:96]

    gen_df["gen_type"] = gen_df["Production Type"].map(TYPE_MAP).fillna("Other")
    gen_df["gen_MW"] = pd.to_numeric(gen_df["Generation (MW)"], errors="coerce").fillna(0)

    gen_pivot = (
        gen_df.groupby(["time", "gen_type"])["gen_MW"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
        .iloc[:96]
    )
    for col in ALL_TYPES:
        if col not in gen_pivot.columns:
            gen_pivot[col] = 0.0

    must_run_mw = gen_pivot[MUST_RUN].sum(axis=1).values
    residual = pd.Series(
        np.maximum(demand.values - must_run_mw, 0),
        index=demand.index,
        name="residual_MW",
    )

    T = len(demand)
    print(f"  Intervals         : {T}  (15-min)")
    print(f"  Avg total demand  : {demand.mean():.0f} MW")
    print(f"  Avg must-run      : {must_run_mw.mean():.0f} MW")
    print(f"  Avg residual      : {residual.mean():.0f} MW")
    print(f"  Must-run sources  : {', '.join(MUST_RUN)}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    demand.to_csv(os.path.join(PROCESSED_DIR, "demand.csv"))
    gen_pivot.to_csv(os.path.join(PROCESSED_DIR, "generation_pivot.csv"))
    residual.to_csv(os.path.join(PROCESSED_DIR, "residual.csv"))
    print(f"\n  Saved to: {PROCESSED_DIR}")

    return demand, gen_pivot, residual


if __name__ == "__main__":
    load_and_process()
