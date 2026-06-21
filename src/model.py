"""
MILP Unit Commitment & Economic Dispatch — DE-LU, 96 × 15-min intervals.

Reads processed data from data/processed/, builds and solves the Pyomo MILP,
then writes to results/:
    dispatch_extended.csv   — MW output per unit per interval
    commitment_extended.csv — binary on/off per unit per interval
    gen_costs.csv           — total generation cost per unit (EUR)
    co2_tonnes.csv          — total CO2-proxy emissions per unit (tonnes)
    summary.json            — headline KPIs

Initial state (warm-start):
    Units in WARM_START are assumed pre-running at t=0 (i.e. they were already
    ON from the previous day). No startup cost is charged at t=0 and their
    min_up obligation is considered satisfied. Edit WARM_START to change which
    units enter the horizon as baseload vs cold.
"""

import json
import os
import sys

import pandas as pd
import pyomo.environ as pyo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

ETS_PRICE = 65.0  # €/tonne CO₂ — EU ETS allowance price (~2025/2026 estimate)

# fuel_mc   : fuel cost + variable O&M only (€/MWh), before ETS allowances
# mc        : computed below as fuel_mc + ETS_PRICE × co2 / 1000
# min_up/dn : minimum up/down time in 15-min intervals (e.g. 24 = 6 hours)
# pmin/pmax : MW  |  su: € startup  |  ramp: MW/15-min  |  co2: kgCO2/MWh
_FLEET = {
    "Lignite": dict(
        pmin=1000,
        pmax=8500,
        fuel_mc=28,
        su=80000,
        ramp=800,
        co2=820,
        min_up=24,
        min_dn=24,
    ),
    "HardCoal": dict(
        pmin=500,
        pmax=9000,
        fuel_mc=38,
        su=50000,
        ramp=900,
        co2=760,
        min_up=16,
        min_dn=16,
    ),
    "CoalGas": dict(
        pmin=100,
        pmax=2000,
        fuel_mc=45,
        su=15000,
        ramp=1500,
        co2=550,
        min_up=8,
        min_dn=4,
    ),
    "Gas_CCGT": dict(
        pmin=300,
        pmax=14000,
        fuel_mc=62,
        su=30000,
        ramp=2000,
        co2=370,
        min_up=8,
        min_dn=4,
    ),
    "Gas_Peak": dict(
        pmin=50, pmax=5000, fuel_mc=98, su=10000, ramp=3500, co2=500, min_up=2, min_dn=2
    ),
    "Oil": dict(
        pmin=50, pmax=1500, fuel_mc=130, su=8000, ramp=2000, co2=650, min_up=4, min_dn=4
    ),
    "Biomass": dict(
        pmin=300, pmax=7000, fuel_mc=18, su=5000, ramp=400, co2=30, min_up=16, min_dn=16
    ),
    "Waste": dict(
        pmin=100,
        pmax=2000,
        fuel_mc=15,
        su=3000,
        ramp=200,
        co2=100,
        min_up=24,
        min_dn=24,
    ),
    "HydroRes": dict(
        pmin=0, pmax=2000, fuel_mc=8, su=500, ramp=2000, co2=0, min_up=1, min_dn=1
    ),
}

GENS = {
    g: {**p, "mc": p["fuel_mc"] + ETS_PRICE * p["co2"] / 1000}
    for g, p in _FLEET.items()
}

# Units assumed to be already running at the start of the horizon (t=0).
# These reflect German baseload plants that operate continuously and would
# not be starting cold on a typical weekday morning. No startup cost is
# charged at t=0 and their min_up obligation from the previous day is
# considered already satisfied.
WARM_START = {"Lignite", "Biomass", "Waste"}

MUST_RUN = [
    "WindOn",
    "WindOff",
    "Solar",
    "HydroRoR",
    "Geothermal",
    "OtherRen",
    "Nuclear",
    "Other",
]
SLACK_PENALTY = 10_000

_STACK_ORDER = [
    "Lignite",
    "HardCoal",
    "CoalGas",
    "Waste",
    "Biomass",
    "HydroRes",
    "Gas_CCGT",
    "Gas_Peak",
    "Oil",
    "PS_Gen",
    "Nuclear",
    "HydroRoR",
    "Geothermal",
    "OtherRen",
    "Other",
    "WindOff",
    "WindOn",
    "Solar",
    "Slack",
]
_REN_COLS = [
    "Solar",
    "WindOn",
    "WindOff",
    "HydroRoR",
    "HydroRes",
    "PS_Gen",
    "Geothermal",
    "OtherRen",
]


def build_and_solve():
    """Build the Pyomo MILP, solve it, and save all results."""
    demand = pd.read_csv(
        os.path.join(PROCESSED_DIR, "demand.csv"), index_col=0, parse_dates=True
    )["demand_MW"]
    residual = pd.read_csv(
        os.path.join(PROCESSED_DIR, "residual.csv"), index_col=0, parse_dates=True
    )["residual_MW"]
    gen_pivot = pd.read_csv(
        os.path.join(PROCESSED_DIR, "generation_pivot.csv"),
        index_col=0,
        parse_dates=True,
    )

    T = len(demand)
    timestamps = demand.index.tolist()

    PS_MAX_GEN = float(gen_pivot["HydroPump"].max()) * 1.1 or 3000
    PS_MAX_PUMP = PS_MAX_GEN * 0.85
    PS_EFF = 0.78
    PS_SOC_MAX = PS_MAX_GEN * 6
    PS_SOC_INIT = PS_SOC_MAX * 0.5

    print(
        f"\nPumped storage  : gen={PS_MAX_GEN:.0f} MW"
        f" | pump={PS_MAX_PUMP:.0f} MW | cap={PS_SOC_MAX:.0f} MWh"
    )

    print("\nBuilding MILP model...")
    mdl = pyo.ConcreteModel(name="UC_ED_DE_LU_Extended")
    mdl.T = pyo.Set(initialize=range(T))
    mdl.G = pyo.Set(initialize=list(GENS.keys()))

    for attr in ("pmin", "pmax", "mc", "su", "ramp"):
        setattr(
            mdl, attr, pyo.Param(mdl.G, initialize={g: GENS[g][attr] for g in GENS})
        )

    mdl.demand = pyo.Param(
        mdl.T, initialize={t: float(residual.iloc[t]) for t in range(T)}
    )

    mdl.p = pyo.Var(mdl.G, mdl.T, within=pyo.NonNegativeReals)
    mdl.u = pyo.Var(mdl.G, mdl.T, within=pyo.Binary)
    mdl.y = pyo.Var(mdl.G, mdl.T, within=pyo.Binary)
    mdl.ps_gen = pyo.Var(mdl.T, within=pyo.NonNegativeReals, bounds=(0, PS_MAX_GEN))
    mdl.ps_pump = pyo.Var(mdl.T, within=pyo.NonNegativeReals, bounds=(0, PS_MAX_PUMP))
    mdl.ps_soc = pyo.Var(mdl.T, within=pyo.NonNegativeReals, bounds=(0, PS_SOC_MAX))
    mdl.ps_bin = pyo.Var(mdl.T, within=pyo.Binary)
    mdl.sv = pyo.Var(mdl.T, within=pyo.NonNegativeReals)
    mdl.z = pyo.Var(
        mdl.G, mdl.T, within=pyo.Binary
    )  # shutdown indicator (1 = OFF at t, ON at t-1)

    def obj_rule(m):
        return (
            sum(m.mc[g] * m.p[g, t] for g in m.G for t in m.T)
            + sum(m.su[g] * m.y[g, t] for g in m.G for t in m.T)
            + sum(SLACK_PENALTY * m.sv[t] for t in m.T)
        )

    mdl.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def c_demand(m, t):
        return (
            sum(m.p[g, t] for g in m.G) + m.ps_gen[t] - m.ps_pump[t] + m.sv[t]
        ) == m.demand[t]

    mdl.c_demand = pyo.Constraint(mdl.T, rule=c_demand)
    mdl.c_pmin = pyo.Constraint(
        mdl.G, mdl.T, rule=lambda m, g, t: m.p[g, t] >= m.pmin[g] * m.u[g, t]
    )
    mdl.c_pmax = pyo.Constraint(
        mdl.G, mdl.T, rule=lambda m, g, t: m.p[g, t] <= m.pmax[g] * m.u[g, t]
    )

    def c_startup(m, g, t):
        if t == 0:
            if g in WARM_START:
                # Unit was pre-running: pin y=0 so no startup cost is charged
                # and the min_up obligation from t<0 is implicitly satisfied.
                return m.y[g, t] == 0
            return m.y[g, t] >= m.u[g, t]
        return m.y[g, t] >= m.u[g, t] - m.u[g, t - 1]

    mdl.c_startup = pyo.Constraint(mdl.G, mdl.T, rule=c_startup)

    def c_ramp_up(m, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        # Big-M relaxation: if unit was OFF at t-1, startup can reach any level ≤ Pmax
        return m.p[g, t] - m.p[g, t - 1] <= m.ramp[g] + m.pmax[g] * (1 - m.u[g, t - 1])

    def c_ramp_dn(m, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        # Big-M relaxation: if unit goes OFF at t, it can drop from any level to zero
        return m.p[g, t - 1] - m.p[g, t] <= m.ramp[g] + m.pmax[g] * (1 - m.u[g, t])

    mdl.c_ramp_up = pyo.Constraint(mdl.G, mdl.T, rule=c_ramp_up)
    mdl.c_ramp_dn = pyo.Constraint(mdl.G, mdl.T, rule=c_ramp_dn)

    # C7: Shutdown indicator  (z[g,t] = 1 when unit transitions ON→OFF at interval t)
    def c_shutdown(m, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.z[g, t] >= m.u[g, t - 1] - m.u[g, t]

    mdl.c_shutdown = pyo.Constraint(mdl.G, mdl.T, rule=c_shutdown)

    # C8: Minimum up-time — once started, must stay ON for min_up intervals
    def c_min_up(m, g, t):
        up = GENS[g]["min_up"]
        if t < up - 1:
            return pyo.Constraint.Skip
        return sum(m.y[g, s] for s in range(t - up + 1, t + 1)) <= m.u[g, t]

    mdl.c_min_up = pyo.Constraint(mdl.G, mdl.T, rule=c_min_up)

    # C9: Minimum down-time — once shut down, must stay OFF for min_dn intervals
    def c_min_dn(m, g, t):
        dn = GENS[g]["min_dn"]
        if t < dn:
            return pyo.Constraint.Skip
        return sum(m.z[g, s] for s in range(t - dn + 1, t + 1)) <= 1 - m.u[g, t]

    mdl.c_min_dn = pyo.Constraint(mdl.G, mdl.T, rule=c_min_dn)

    def c_soc(m, t):
        prev = PS_SOC_INIT if t == 0 else m.ps_soc[t - 1]
        return m.ps_soc[t] == prev + PS_EFF * m.ps_pump[t] / 4 - m.ps_gen[t] / 4

    mdl.c_soc = pyo.Constraint(mdl.T, rule=c_soc)
    mdl.c_ps_gen = pyo.Constraint(
        mdl.T, rule=lambda m, t: m.ps_gen[t] <= PS_MAX_GEN * m.ps_bin[t]
    )
    mdl.c_ps_pump = pyo.Constraint(
        mdl.T, rule=lambda m, t: m.ps_pump[t] <= PS_MAX_PUMP * (1 - m.ps_bin[t])
    )

    n_vars = sum(1 for _ in mdl.component_data_objects(pyo.Var))
    n_con = sum(1 for _ in mdl.component_data_objects(pyo.Constraint))
    print(f"  Variables   : {n_vars:,}")
    print(f"  Constraints : {n_con:,}")

    print("\nSolving ...")
    if pyo.SolverFactory("gurobi").available():
        solver = pyo.SolverFactory("gurobi")
        print("  Using Gurobi")
        sol = solver.solve(mdl, tee=False, options={"TimeLimit": 120, "MIPGap": 0.01})
    elif pyo.SolverFactory("appsi_highs").available():
        solver = pyo.SolverFactory("appsi_highs")
        print("  Using HiGHS (pip install highspy)")
        sol = solver.solve(mdl, tee=False)
    elif pyo.SolverFactory("cbc").available():
        solver = pyo.SolverFactory("cbc")
        print("  Using CBC")
        sol = solver.solve(mdl, tee=False, options={"seconds": 120, "ratio": 0.01})
    else:
        print("\n  No solver found.  pip install highspy")
        sys.exit(1)

    print(f"  Solver status : {sol.solver.termination_condition}")

    # Extract results
    dispatch = pd.DataFrame(index=timestamps)
    commit = pd.DataFrame(index=timestamps)

    for g in GENS:
        dispatch[g] = [pyo.value(mdl.p[g, t]) for t in range(T)]
        commit[g] = [round(pyo.value(mdl.u[g, t])) for t in range(T)]

    for src in MUST_RUN:
        dispatch[src] = gen_pivot[src].values[:T]

    dispatch["PS_Gen"] = [pyo.value(mdl.ps_gen[t]) for t in range(T)]
    dispatch["PS_Pump"] = [pyo.value(mdl.ps_pump[t]) for t in range(T)]
    dispatch["PS_SoC"] = [pyo.value(mdl.ps_soc[t]) for t in range(T)]
    dispatch["Slack"] = [pyo.value(mdl.sv[t]) for t in range(T)]

    total_cost = pyo.value(mdl.obj)
    gen_costs = {
        g: sum(GENS[g]["mc"] * pyo.value(mdl.p[g, t]) / 4 for t in range(T))
        for g in GENS
    }
    co2_tonnes = {
        g: sum(GENS[g]["co2"] * pyo.value(mdl.p[g, t]) / 1000 / 4 for t in range(T))
        for g in GENS
    }

    total_gen = (
        dispatch[[g for g in _STACK_ORDER if g in dispatch.columns and g != "Slack"]]
        .clip(lower=0)
        .sum()
        .sum()
        / 4
    )
    ren_gen = (
        dispatch[[c for c in _REN_COLS if c in dispatch.columns]]
        .clip(lower=0)
        .sum()
        .sum()
        / 4
    )
    n_starts = sum(round(pyo.value(mdl.y[g, t])) for g in GENS for t in range(T))

    summary = {
        "total_cost_eur": total_cost,
        "total_gen_MWh": total_gen,
        "ren_gen_MWh": ren_gen,
        "ren_share_pct": 100 * ren_gen / total_gen,
        "avg_marginal_cost": total_cost / total_gen,
        "total_co2_tonnes": sum(co2_tonnes.values()),
        "n_startups": n_starts,
        "max_slack_MW": float(dispatch["Slack"].max()),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dispatch.to_csv(os.path.join(RESULTS_DIR, "dispatch_extended.csv"))
    commit.to_csv(os.path.join(RESULTS_DIR, "commitment_extended.csv"))
    pd.Series(gen_costs, name="cost_eur").to_csv(
        os.path.join(RESULTS_DIR, "gen_costs.csv")
    )
    pd.Series(co2_tonnes, name="co2_tonnes").to_csv(
        os.path.join(RESULTS_DIR, "co2_tonnes.csv")
    )
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Results saved to: {RESULTS_DIR}")
    _print_summary(summary, gen_costs)

    return dispatch, commit, summary


def _print_summary(summary, gen_costs):
    print("\n" + "=" * 55)
    print("SUMMARY — DE-LU, 10 May 2026")
    print("=" * 55)
    print(f"  Total energy served : {summary['total_gen_MWh'] / 1000:.1f} GWh")
    print(f"  Renewable share     : {summary['ren_share_pct']:.1f}%")
    print(f"  Total system cost   : {summary['total_cost_eur']:,.0f} EUR")
    print(f"  Avg marginal cost   : {summary['avg_marginal_cost']:.2f} EUR/MWh")
    print(f"  Total CO₂ proxy     : {summary['total_co2_tonnes'] / 1000:.0f} kt")
    print(f"  Total unit startups : {summary['n_startups']}")
    print(f"  Max slack (MW)      : {summary['max_slack_MW']:.1f}")
    print("=" * 55)
    print("\nCost breakdown:")
    for g, c in sorted(gen_costs.items(), key=lambda x: -x[1]):
        share = 100 * c / max(sum(gen_costs.values()), 1)
        print(f"  {g:<12} {c / 1000:>7.0f} k€  ({share:.1f}%)")


if __name__ == "__main__":
    build_and_solve()
