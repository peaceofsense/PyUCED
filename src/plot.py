"""
Visualisation for DE-LU MILP UC/ED results.

Reads from results/ and data/processed/, produces a 4-panel figure
saved to plots/dispatch_extended.png.

Panels:
  1. Full stacked generation dispatch vs demand
  2. Pumped-storage power flows and state of charge
  3. Cost vs CO2-proxy breakdown by unit
  4. Unit commitment heatmap
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "plots")

COLORS = {
    "Lignite":    "#3d2b1f",
    "HardCoal":   "#5a5a5a",
    "CoalGas":    "#8c7b6e",
    "Gas_CCGT":   "#e88c2a",
    "Gas_Peak":   "#f5c842",
    "Oil":        "#c0392b",
    "Biomass":    "#5cad4a",
    "Waste":      "#a07850",
    "HydroRes":   "#2471a3",
    "PS_Gen":     "#1a5276",
    "Nuclear":    "#9b59b6",
    "HydroRoR":   "#4a90d9",
    "Geothermal": "#e67e22",
    "WindOn":     "#89cff0",
    "WindOff":    "#5b9bd5",
    "Solar":      "#ffd966",
    "OtherRen":   "#82e0aa",
    "Other":      "#aaa",
    "Slack":      "#ff4444",
}

STACK_ORDER = [
    "Lignite", "HardCoal", "CoalGas", "Waste", "Biomass", "HydroRes",
    "Gas_CCGT", "Gas_Peak", "Oil", "PS_Gen", "Nuclear", "HydroRoR",
    "Geothermal", "OtherRen", "Other", "WindOff", "WindOn", "Solar", "Slack",
]

DISP_GENS = [
    "Lignite", "HardCoal", "CoalGas", "Gas_CCGT", "Gas_Peak",
    "Oil", "Biomass", "Waste", "HydroRes",
]


def make_plots():
    """Load results and processed data, render and save the 4-panel figure."""
    demand = pd.read_csv(
        os.path.join(PROCESSED_DIR, "demand.csv"), index_col=0, parse_dates=True
    )["demand_MW"]
    residual = pd.read_csv(
        os.path.join(PROCESSED_DIR, "residual.csv"), index_col=0, parse_dates=True
    )["residual_MW"]
    dispatch = pd.read_csv(
        os.path.join(RESULTS_DIR, "dispatch_extended.csv"), index_col=0, parse_dates=True
    )
    commit = pd.read_csv(
        os.path.join(RESULTS_DIR, "commitment_extended.csv"), index_col=0, parse_dates=True
    )
    gen_costs  = pd.read_csv(
        os.path.join(RESULTS_DIR, "gen_costs.csv"), index_col=0
    )["cost_eur"].to_dict()
    co2_tonnes = pd.read_csv(
        os.path.join(RESULTS_DIR, "co2_tonnes.csv"), index_col=0
    )["co2_tonnes"].to_dict()

    T = len(demand)
    x = np.arange(T)
    timestamps = demand.index.tolist()

    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(
        "Extended MILP Unit Commitment & Economic Dispatch\n"
        "DE-LU | 10 May 2026 | 15-min resolution | All ENTSO-E Generation Types",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1 — Stacked dispatch
    ax = axes[0]
    bottom = np.zeros(T)
    for g in STACK_ORDER:
        if g in dispatch.columns:
            vals = dispatch[g].clip(lower=0).values
            ax.bar(x, vals, bottom=bottom, color=COLORS.get(g, "#aaa"),
                   label=g, width=1.0, alpha=0.92)
            bottom += vals
    ax.plot(x, demand.values,   color="black", lw=2,       label="Total demand",    zorder=5)
    ax.plot(x, residual.values, color="red",   lw=1, ls="--", label="Residual demand", zorder=5)
    ax.set_ylabel("Power (MW)")
    ax.set_title("Full Generation Dispatch Stack (all technologies)")
    ax.legend(loc="upper right", ncol=5, fontsize=7)
    ax.set_ylim(0, demand.max() * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f} GW"))

    # Panel 2 — Pumped storage SoC and flows
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(x,  dispatch["PS_Gen"].values,  color="#1a5276", width=1.0, alpha=0.8, label="PS discharge")
    ax.bar(x, -dispatch["PS_Pump"].values, color="#a9cce3", width=1.0, alpha=0.8, label="PS charge")
    ax2.plot(x, dispatch["PS_SoC"].values, color="#117a65", lw=2, label="SoC (MWh)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Power (MW)")
    ax2.set_ylabel("State of Charge (MWh)", color="#117a65")
    ax.set_title("Pumped Storage: Dispatch & State of Charge")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)

    # Panel 3 — Cost vs CO2 per unit
    ax = axes[2]
    ax2 = ax.twinx()
    labels = list(gen_costs.keys())
    costs  = [gen_costs[g] / 1000 for g in labels]
    co2s   = [co2_tonnes[g] / 1000 for g in labels]
    bar_w  = 0.35
    pos    = np.arange(len(labels))
    b1 = ax.bar(
        pos - bar_w / 2, costs, bar_w,
        color=[COLORS.get(g, "#aaa") for g in labels], edgecolor="white", label="Cost (k€)",
    )
    b2 = ax2.bar(
        pos + bar_w / 2, co2s, bar_w,
        color=[COLORS.get(g, "#aaa") for g in labels], edgecolor="white", alpha=0.5, label="CO₂ (kt)",
    )
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Generation Cost (k€)")
    ax2.set_ylabel("CO₂ Emissions (kt)", color="#555")
    ax.set_title("Cost vs CO₂ Proxy by Unit")
    ax.legend([b1, b2], ["Cost (k€)", "CO₂ (kt)"], fontsize=8)

    # Panel 4 — Commitment heatmap
    ax = axes[3]
    cm = commit[DISP_GENS].T.values.astype(float)
    im = ax.imshow(cm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="none")
    ax.set_yticks(range(len(DISP_GENS)))
    ax.set_yticklabels(DISP_GENS, fontsize=9)
    ax.set_title("Unit Commitment Status  (Green = ON,  Red = OFF)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    # Shared x-axis
    tick_pos    = list(range(0, T, 4))
    tick_labels = [timestamps[i].strftime("%H:%M") for i in tick_pos]
    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_png = os.path.join(PLOTS_DIR, "dispatch_extended.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {out_png}")


if __name__ == "__main__":
    make_plots()
