# Day-Ahead MILP Unit Commitment & Economic Dispatch — Germany (DE-LU)

A Mixed-Integer Linear Program (MILP) that answers:

> *"Which power plants should run, and at what output level, to meet Germany's electricity demand on 10 May 2026 at minimum cost?"*

Built with real ENTSO-E market data, [Pyomo](http://www.pyomo.org/), and the open-source [HiGHS](https://highs.dev/) solver. Covers 96 × 15-minute intervals (one full day) across all major generation technologies in the DE-LU bidding zone.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quickstart](#quickstart)
3. [What This Model Does](#what-this-model-does)
4. [Mathematical Formulation](#mathematical-formulation)
5. [What This Model Does NOT Do](#what-this-model-does-not-do)
6. [Data Sources](#data-sources)
7. [Assumed Parameters](#assumed-parameters)
8. [Outputs](#outputs)
9. [Dependencies](#dependencies)

---

## Project Structure

```
.
├── main.py                        # Pipeline runner (data → model → plots)
├── src/
│   ├── data_processing.py         # Load & clean raw ENTSO-E CSVs
│   ├── model.py                   # Build & solve the Pyomo MILP
│   └── plot.py                    # Generate 4-panel results figure
├── data/
│   ├── raw/                       # Original ENTSO-E CSV exports (input)
│   └── processed/                 # Cleaned outputs from data_processing.py
├── results/                       # Solver outputs (CSVs + summary JSON)
└── plots/                         # Generated figures
```

---

## Quickstart

**1. Install dependencies (once):**

```bash
pip install pyomo highspy pandas matplotlib numpy
```

**2. Place raw data files in `data/raw/`:**

```
GUI_TOTAL_LOAD_DAYAHEAD_202605092200-202605102200.csv
AGGREGATED_GENERATION_PER_TYPE_GENERATION_202605092200-202605102200.csv
```

**3. Run the full pipeline:**

```bash
python main.py
```

Each stage can also be run independently:

```bash
python src/data_processing.py   # step 1: clean data only
python src/model.py             # step 2: solve MILP only (requires step 1)
python src/plot.py              # step 3: regenerate plots only (requires step 2)
```

---

## What This Model Does

### Problem Type
This is a **Unit Commitment (UC) + Economic Dispatch (ED)** problem, a standard formulation in power systems operations. It simultaneously decides:

- **Commitment** — which generators to switch on or off at each time step (binary decision)
- **Dispatch** — how much power each committed generator should produce (continuous decision)

The objective is to minimize total operating cost (variable generation cost + startup costs) while serving the forecast demand at every interval.

### Time Horizon
- **Date:** 10 May 2026
- **Resolution:** 15 minutes per interval
- **Intervals:** 96 (covering the full 24-hour day)
- **Timezone:** CET/CEST (as reported by ENTSO-E)

### Bidding Zone
- **DE-LU** — the Germany/Luxembourg electricity market area (as defined by ENTSO-E)

### Technology Treatment

The model splits all generation into three categories:

#### 1. Must-Run Fleet (taken directly from ENTSO-E data, not optimized)
These technologies are treated as fixed injections — their output is read from the actual generation data and subtracted from total demand to form the **residual demand** that the MILP must serve.

| Internal Label | ENTSO-E Source Type(s) |
|---|---|
| `WindOn` | Wind Onshore |
| `WindOff` | Wind Offshore |
| `Solar` | Solar |
| `HydroRoR` | Hydro Run-of-river and pondage |
| `Geothermal` | Geothermal |
| `OtherRen` | Other renewable |
| `Nuclear` | Nuclear |
| `Other` | Other, Fossil Peat, Marine |

**Why must-run?** Wind, solar, and run-of-river hydro cannot be freely dispatched — their output is determined by weather and hydrology. Nuclear and geothermal have very high startup costs and are typically base-loaded. Treating them as fixed simplifies the model while remaining realistic for a single-day horizon.

#### 2. Dispatchable Fleet (MILP-optimized)
These nine technology classes are jointly optimized with binary commitment variables, minimum/maximum output bounds, ramp rate limits, and startup costs.

| Internal Label | ENTSO-E Source Type(s) |
|---|---|
| `Lignite` | Fossil Brown coal/Lignite |
| `HardCoal` | Fossil Hard coal |
| `CoalGas` | Fossil Coal-derived gas |
| `Gas_CCGT` | Fossil Gas (combined-cycle) |
| `Gas_Peak` | Peaking gas (open-cycle, assumed) |
| `Oil` | Fossil Oil, Fossil Oil shale |
| `Biomass` | Biomass |
| `Waste` | Waste |
| `HydroRes` | Hydro Water Reservoir |

#### 3. Pumped Hydro Storage (SoC dynamics modeled)
Pumped hydro is modeled explicitly with:
- Separate generation and pumping power variables
- State of Charge (SoC) tracked across all 96 intervals
- A binary mode variable preventing simultaneous generation and pumping
- Parameters derived from the observed maximum output in the ENTSO-E generation data (see [Assumed Parameters](#assumed-parameters))

---

## Mathematical Formulation

### Sets
| Symbol | Description |
|---|---|
| `T = {0, 1, ..., 95}` | Time intervals (15-min each) |
| `G` | Dispatchable generators: {Lignite, HardCoal, CoalGas, Gas_CCGT, Gas_Peak, Oil, Biomass, Waste, HydroRes} |

### Decision Variables

| Variable | Type | Description |
|---|---|---|
| `p[g,t]` | Continuous ≥ 0 | Dispatch output of unit `g` at interval `t` (MW) |
| `u[g,t]` | Binary | Commitment status of unit `g` at `t` (1 = ON) |
| `y[g,t]` | Binary | Startup indicator (1 = unit starts at interval `t`) |
| `ps_gen[t]` | Continuous ≥ 0 | Pumped storage generation (MW) |
| `ps_pump[t]` | Continuous ≥ 0 | Pumped storage pumping load (MW) |
| `ps_soc[t]` | Continuous ≥ 0 | Pumped storage state of charge (MWh) |
| `ps_bin[t]` | Binary | PS mode: 1 = generating, 0 = pumping or idle |
| `sv[t]` | Continuous ≥ 0 | Slack variable (load-shedding proxy, heavily penalised) |
| `z[g,t]` | Binary | Shutdown indicator (1 = unit transitions ON→OFF at interval `t`) |

### Objective Function

Minimize total system cost:

```
min  Σ_{g,t} mc[g] · p[g,t]          (variable generation cost)
   + Σ_{g,t} su[g] · y[g,t]          (startup cost)
   + Σ_t     10,000 · sv[t]           (slack penalty)
```

The slack penalty (10,000 €/MWh) is far above any generator marginal cost, so the solver only uses it when the residual demand cannot be served by the available fleet — it should be zero in a feasible solution.

### Constraints

**C1 — Demand balance** (residual demand = dispatchable output ± pumped storage + slack):
```
Σ_g p[g,t]  +  ps_gen[t]  -  ps_pump[t]  +  sv[t]  =  residual[t]    ∀ t
```

**C2 — Minimum output** (generator cannot produce below pmin if committed):
```
p[g,t]  ≥  pmin[g] · u[g,t]    ∀ g, t
```

**C3 — Maximum output** (generator cannot exceed pmax):
```
p[g,t]  ≤  pmax[g] · u[g,t]    ∀ g, t
```

**C4 — Startup logic** (y[g,t] = 1 whenever unit transitions from OFF to ON):
```
y[g,t]  ≥  u[g,t] - u[g,t-1]    ∀ g, t > 0
y[g,0]  ≥  u[g,0]                ∀ g ∉ WARM_START    (cold-start: pay startup if committed at t=0)
y[g,0]  =  0                     ∀ g ∈ WARM_START     (pre-running: no startup cost at t=0)
```
See [Initial Unit State (Warm-Start)](#initial-unit-state-warm-start) for which units are in `WARM_START`.

**C5 — Ramp-up limit** (big-M relaxation at startup):
```
p[g,t] - p[g,t-1]  ≤  ramp[g] + Pmax[g]·(1−u[g,t−1])    ∀ g, t > 0
```

**C6 — Ramp-down limit** (big-M relaxation at shutdown):
```
p[g,t-1] - p[g,t]  ≤  ramp[g] + Pmax[g]·(1−u[g,t])      ∀ g, t > 0
```

**C7 — Shutdown indicator** (z[g,t] = 1 whenever unit goes from ON at t-1 to OFF at t):
```
z[g,t]  ≥  u[g,t-1] - u[g,t]    ∀ g, t > 0
```

**C8 — Minimum up-time** (once started, must remain ON for `min_up` intervals):
```
Σ_{s=t-min_up+1}^{t} y[g,s]  ≤  u[g,t]    ∀ g, t ≥ min_up - 1
```
If any startup occurred in the last `min_up` intervals, the unit must be ON now.

**C9 — Minimum down-time** (once shut down, must remain OFF for `min_dn` intervals):
```
Σ_{s=t-min_dn+1}^{t} z[g,s]  ≤  1 - u[g,t]    ∀ g, t ≥ min_dn
```
If any shutdown occurred in the last `min_dn` intervals, the unit must be OFF now.

**C10 — Pumped storage SoC dynamics** (energy balance per interval):
```
ps_soc[t]  =  ps_soc[t-1]  +  η · ps_pump[t]/4  -  ps_gen[t]/4    ∀ t
```
where `η = 0.78` is the pumping efficiency and `/4` converts 15-min power to energy (MWh).  
Initial SoC: `ps_soc[-1] = PS_SOC_INIT = PS_SOC_MAX × 0.5`

**C11 — Pumped storage cannot generate while pumping**:
```
ps_gen[t]   ≤  PS_MAX_GEN  · ps_bin[t]        ∀ t
ps_pump[t]  ≤  PS_MAX_PUMP · (1 - ps_bin[t])  ∀ t
```

---

## What This Model Does NOT Do

These are deliberate simplifications. Understanding them is important for interpreting results.

| Limitation | Explanation |
|---|---|
| **No transmission network** | The model is a single-node "copper-plate" model — it assumes Germany's grid has no internal congestion and all generators can freely serve all demand. Real market prices vary by region (nodal or zonal pricing). |
| **No cross-border flows** | Imports and exports via interconnectors (France, Denmark, Poland, etc.) are ignored. In reality DE-LU is a net importer or exporter depending on the hour, which affects dispatch. |
| **Minimum up/down times are approximate** | Minimum run and off times are included (e.g. 6 hours for lignite, 4 hours for hard coal) but are representative estimates. Real plant-level values vary by unit age, cold/warm/hot start state, and regulatory constraints. |
| **No spinning or operating reserves** | Real dispatch includes reserve requirements (primary, secondary, tertiary). The model only enforces exact demand balance; it does not hold capacity back for uncertainty. |
| **ETS price is a fixed assumption** | EU ETS allowance cost is included in marginal costs at a fixed `ETS_PRICE = 65 €/tonne`. Real ETS prices fluctuate daily. Changing this value in `src/model.py` will shift the coal-vs-gas dispatch crossover point. |
| **Warm-start units use a simplified initial state** | Lignite, Biomass, and Waste are assumed pre-running at midnight (`WARM_START`). No startup cost is charged at `t=0` and their min_up obligation from the prior day is treated as satisfied. Cold-start units (Gas, HardCoal, etc.) must pay startup cost if committed at `t=0`. Edit `WARM_START` in `src/model.py` to change this assumption. |
| **Generator parameters are representative estimates** | `pmin`, `pmax`, `mc`, `su`, `ramp`, and `co2` values are literature-based approximations for Germany's fleet, not calibrated against actual plant-level data from BNetzA or company reports (see [Assumed Parameters](#assumed-parameters)). |
| **Nuclear treated as must-run** | Nuclear is fixed to its observed generation profile. In reality it is partially dispatchable, but this simplification is appropriate for a single-day model. |
| **Pumped storage parameters partially derived** | `PS_MAX_GEN` is set to 110% of the maximum observed HydroPump generation in the ENTSO-E data. The energy capacity and efficiency are assumed (see below). |
| **Single day, no multi-day storage optimisation** | The model's horizon is exactly 24 hours. Pumped storage SoC does not carry over across days and is not optimised with a look-ahead beyond the horizon. |
| **No demand response or flexible demand** | Industrial curtailment, EV charging flexibility, and demand-side management are not modeled. |
| **No intraday or balancing market** | The model reflects a stylised day-ahead market clearing only. Balancing actions and redispatch are not captured. |
| **Fossil Gas mapped entirely to Gas_CCGT** | The ENTSO-E "Fossil Gas" category includes both CCGT and OCGT plants. This model treats the entire class as CCGT (high efficiency, lower cost), which may underestimate peaking costs. A separate `Gas_Peak` category is included with assumed OCGT parameters. |
| **No end-of-day storage terminal constraint** | Pumped storage SoC is free to reach any level by the final interval. The solver will tend to fully discharge storage toward midnight to reduce cost — a classic horizon-gaming artifact. A real model would enforce `ps_soc[95] ≥ ps_soc_init` or optimise across multiple days. |
| **Flat marginal cost (no part-load efficiency)** | Each generator's marginal cost is constant regardless of output level. Real thermal plants have higher heat rates at part load, so true MC increases as output falls below the design point. This simplification is standard in aggregated UC models but overstates efficiency at low dispatch levels. |
| **One aggregated unit per technology class** | Germany's grid contains hundreds of individual plants. Modeling each technology as a single unit means the binary commitment variable has no physical plant-level meaning and the pmin/pmax bounds represent the entire fleet, not individual unit constraints. |
| **`avg_marginal_cost` in summary.json is not the system marginal price** | The reported value is `total_variable_cost / total_energy_served` — a cost-weighted average, not the shadow price of the demand constraint. Real UC models extract dual variables (locational marginal prices / LMPs) from the LP relaxation at the solved commitment. This model does not compute LMPs. |
| **Day-ahead actual load used, not a forecast** | Input load is taken from the ENTSO-E "Actual Total Load" column, not a day-ahead forecast. A real day-ahead market clears on forecasts; using actuals gives the model slightly better information than operators would have had at the time of clearing. |

---

## Data Sources

### Load Data
- **Source:** [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — *Total Load — Day Ahead / Actual*
- **Area:** DE-LU (Germany / Luxembourg bidding zone)
- **Period:** 09 May 2026 22:00 CET → 10 May 2026 22:00 CET (covers full delivery day)
- **File:** `GUI_TOTAL_LOAD_DAYAHEAD_202605092200-202605102200.csv`
- **Column used:** `Actual Total Load (MW)`
- **Intervals used:** First 96 rows after parsing and sorting by timestamp

### Generation Data
- **Source:** [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — *Actual Generation per Production Type*
- **Area:** DE-LU
- **Period:** Same as above
- **File:** `AGGREGATED_GENERATION_PER_TYPE_GENERATION_202605092200-202605102200.csv`
- **Column used:** `Generation (MW)` grouped by `Production Type`
- **Intervals used:** First 96 rows per type after pivot and sorting

### ENTSO-E Production Type Mapping

The raw ENTSO-E labels are mapped to internal model labels as follows:

| ENTSO-E Label | Internal Label | Treatment |
|---|---|---|
| Wind Onshore | `WindOn` | Must-run |
| Wind Offshore | `WindOff` | Must-run |
| Solar | `Solar` | Must-run |
| Hydro Run-of-river and pondage | `HydroRoR` | Must-run |
| Geothermal | `Geothermal` | Must-run |
| Other renewable | `OtherRen` | Must-run |
| Nuclear | `Nuclear` | Must-run |
| Other | `Other` | Must-run |
| Fossil Peat | `Other` | Must-run (merged into Other) |
| Marine | `Other` | Must-run (merged into Other) |
| Fossil Brown coal/Lignite | `Lignite` | Dispatchable |
| Fossil Hard coal | `HardCoal` | Dispatchable |
| Fossil Coal-derived gas | `CoalGas` | Dispatchable |
| Fossil Gas | `Gas_CCGT` | Dispatchable |
| Fossil Oil | `Oil` | Dispatchable |
| Fossil Oil shale | `Oil` | Dispatchable (merged into Oil) |
| Biomass | `Biomass` | Dispatchable |
| Waste | `Waste` | Dispatchable |
| Hydro Water Reservoir | `HydroRes` | Dispatchable |
| Hydro Pumped Storage | `HydroPump` | Storage (SoC model) |
| Energy storage | `HydroPump` | Storage (merged into HydroPump) |

---

## Assumed Parameters

These values are **representative estimates** based on publicly available literature for the German power market. They are **not** calibrated against actual plant-level data (e.g. BNetzA power plant list, company filings). Results should be interpreted as illustrative, not as forecasts of actual market dispatch.

### Dispatchable Generator Fleet

ETS allowance cost (`ETS_PRICE = 65 €/tonne`) is added to each generator's fuel marginal cost. This reflects the real cash cost German generators face when purchasing EU ETS allowances. At 65 €/tonne, Gas CCGT (86 €/MWh) cleanly dispatches before Hard Coal (87 €/MWh) — the real-world "fuel switching" observed in the German market when ETS prices are elevated. Lignite remains cheaper than gas until ETS reaches ~90–100 €/tonne, which is the policy threshold for full coal phase-out.

| Unit | Pmin (MW) | Pmax (MW) | Fuel mc (€/MWh) | ETS @65 (€/MWh) | **Total mc (€/MWh)** | Startup (€) | Ramp (MW/15-min) | Min Up (intervals) | Min Down (intervals) | CO₂ (kg/MWh) |
|---|---|---|---|---|---|---|---|---|---|---|
| Hydro Reservoir | 0 | 2,000 | 8 | 0 | **8** | 500 | 2,000 | 1 | 1 | 0 |
| Biomass | 300 | 7,000 | 18 | 2.0 | **20** | 5,000 | 400 | 16 | 16 | 30 |
| Waste | 100 | 2,000 | 15 | 6.5 | **21.5** | 3,000 | 200 | 24 | 24 | 100 |
| Coal-derived Gas | 100 | 2,000 | 45 | 35.8 | **80.8** | 15,000 | 1,500 | 8 | 4 | 550 |
| Lignite | 1,000 | **8,500** | 28 | 53.3 | **81.3** | 80,000 | 800 | 24 | 24 | 820 |
| Gas CCGT | 300 | 14,000 | 62 | 24.1 | **86.1** | 30,000 | 2,000 | 8 | 4 | 370 |
| Hard Coal | 500 | 9,000 | 38 | 49.4 | **87.4** | 50,000 | 900 | 16 | 16 | 760 |
| Gas Peaker (OCGT) | 50 | 5,000 | 98 | 32.5 | **130.5** | 10,000 | 3,500 | 2 | 2 | 500 |
| Oil | 50 | 1,500 | 130 | 42.3 | **172.3** | 8,000 | 2,000 | 4 | 4 | 650 |

*Table sorted by total mc (merit order). Min Up/Down are in 15-min intervals — e.g. 24 = 6 hours. Lignite Pmax reduced to 8,500 MW to reflect actual DE installed capacity post phase-out closures.*

**Notes on specific assumptions:**

- **Pmax values** represent the assumed total installed capacity available for each technology class in Germany, not any single plant. Lignite (8.5 GW) and Gas CCGT (14 GW) reflect approximately the installed fleet sizes as of 2025/2026 after several coal phase-out steps.
- **Marginal costs** = fuel cost + variable O&M + ETS allowance cost (`ETS_PRICE × co2 / 1000`). Fuel prices assumed: gas ~30 €/MWh, coal ~10 €/MWh, lignite ~3 €/MWh (mid-2026 estimates). ETS set at 65 €/tonne — adjust `ETS_PRICE` in `src/model.py` to test sensitivity (e.g. ETS ~90–100 €/t is where lignite becomes more expensive than gas CCGT).
- **Lignite Pmax** is set to 8,500 MW reflecting Germany's actual installed lignite capacity post phase-out closures (~2025/2026). The original 17,000 MW over-represented available capacity and caused lignite to dominate the dispatch stack unrealistically.
- **Minimum up/down times** prevent unrealistic single-interval cycling. Values are in 15-min intervals: lignite and waste (24 = 6 h), hard coal and biomass (16 = 4 h), CCGT and coal gas (8 = 2 h up, 4 = 1 h down), gas peaker (2 = 30 min), oil (4 = 1 h), hydro reservoir (1 = 15 min).
- **Startup costs** are aggregate fleet-level estimates. Real startup costs depend on the number of units started and their cold/warm/hot state.
- **Ramp rates** are per 15-minute interval. Gas peakers have the highest flexibility (3,500 MW/15-min), lignite the lowest (800 MW/15-min), consistent with thermal inertia characteristics.
- **CO₂ intensity** values follow IPCC 2014 / EEA emission factor estimates for gross calorific value. The CO₂ calculation in results is a proxy (does not account for efficiency curves or part-load penalties).
- **Biomass CO₂** is set to 30 kg/MWh as a biogenic accounting convention (near-zero on a lifecycle basis under IPCC accounting, though the actual combustion intensity is higher).

### Pumped Hydro Storage

| Parameter | Value | Derivation |
|---|---|---|
| Max generation power (`PS_MAX_GEN`) | `max(observed HydroPump output) × 1.1` | 10% headroom above observed peak in ENTSO-E data, or 3,000 MW if data is zero |
| Max pumping power (`PS_MAX_PUMP`) | `PS_MAX_GEN × 0.85` | Pumps are typically rated slightly below turbine capacity |
| Pumping efficiency (`η`) | 0.78 | Round-trip efficiency applied on the pump side; typical for older German pumped hydro fleet |
| Energy capacity (`PS_SOC_MAX`) | `PS_MAX_GEN × 6` MWh | Assumes ~6 hours of full-power discharge, consistent with published capacity for German pumped hydro (~40 GWh total fleet) |
| Initial state of charge | `PS_SOC_MAX × 0.5` | 50% full at midnight — a neutral starting assumption |

### Initial Unit State (Warm-Start)

Single-day UC models face a **cold-start problem**: if all units are assumed OFF at `t=0`, baseload generators like lignite must pay their full startup cost each day and commit for their minimum up-time before being allowed to turn off. This is unrealistic — German lignite and biomass plants are typically baseload and were already running continuously the day before.

The `WARM_START` set in `src/model.py` solves this by pinning the startup indicator `y[g,0] = 0` for pre-running units. This means:

| Effect | Explanation |
|---|---|
| No startup cost at `t=0` | The 80,000 € lignite startup cost is not charged — it was paid on a previous day |
| No forced min_up lock-in from `t=0` | Min_up from the prior day is considered already satisfied; unit can turn off freely within the horizon |
| Normal commitment constraints from `t=1` onwards | If the unit shuts down and restarts within the horizon, full startup cost and min_up/min_dn apply |

| Unit | Warm-Start? | Reasoning |
|---|---|---|
| Lignite | ✓ Yes | Baseload in Germany; runs 24/7, rarely cold-starts |
| Biomass | ✓ Yes | Near-baseload; high fuel cost if cycled, runs continuously |
| Waste | ✓ Yes | Waste-to-energy plants are contractually baseload; cannot easily cycle |
| HardCoal | ✗ No | Mid-merit; starts cold on days with moderate demand |
| Gas_CCGT | ✗ No | Flexible mid-merit/peaker; frequently cycles daily |
| Gas_Peak | ✗ No | Peaker; only starts during high-demand periods |
| CoalGas | ✗ No | Flexible intermediate unit |
| Oil | ✗ No | Emergency/peaker; rarely running |
| HydroRes | ✗ No | Flexible dispatchable; dispatched on merit per interval |

### Slack Penalty

The slack variable `sv[t]` represents unserved energy. Its penalty coefficient is set to **10,000 €/MWh**, which is approximately 59× above the highest generator marginal cost (Oil at 169 €/MWh including ETS). This ensures the optimiser exhausts all generation options before shedding load. In a correctly parameterized model, slack should be zero in all intervals.

---

## Outputs

After running `python main.py`, the following files are created:

### `data/processed/`
| File | Description |
|---|---|
| `demand.csv` | 96-row time series of actual total load (MW), indexed by timestamp |
| `generation_pivot.csv` | 96 × 17 table of generation by technology class (MW), indexed by timestamp |
| `residual.csv` | 96-row time series of residual demand = total load − must-run generation (MW) |

### `results/`
| File | Description |
|---|---|
| `dispatch_extended.csv` | Full dispatch schedule: MW output per technology per 15-min interval. Columns include all 9 dispatchable units + 8 must-run sources + PS_Gen, PS_Pump, PS_SoC, Slack |
| `commitment_extended.csv` | Binary on/off matrix for the 9 dispatchable units across all 96 intervals |
| `gen_costs.csv` | Total generation variable cost per unit over the full day (EUR) |
| `co2_tonnes.csv` | Total CO₂-proxy emissions per unit over the full day (tonnes) |
| `summary.json` | Headline KPIs: total cost, total energy served, renewable share, avg marginal cost, CO₂ total, number of startups, max slack |

### `plots/`
| File | Description |
|---|---|
| `dispatch_extended.png` | 4-panel figure: (1) stacked dispatch vs demand, (2) pumped storage SoC and flows, (3) cost vs CO₂ by unit, (4) unit commitment heatmap |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyomo` | MILP model formulation |
| `highspy` | HiGHS solver interface (recommended) |
| `pandas` | Data loading, processing, and CSV I/O |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |

**Solver fallback order:** HiGHS → Gurobi → CBC. HiGHS (free, via `pip install highspy`) is recommended and sufficient for this problem size (~4,000 variables, ~10,000 constraints). Solve time is approximately 5–30 seconds on a modern laptop.

```bash
pip install pyomo highspy pandas matplotlib numpy
```

---

## Limitations Summary for Academic / Portfolio Context

This model is built as an academic portfolio project demonstrating MILP formulation for power systems. It is appropriate for:
- Illustrating unit commitment and economic dispatch methodology
- Understanding merit order and generator dispatch stacking
- Exploring pumped storage scheduling under SoC constraints
- Sensitivity analysis on marginal costs, renewable penetration, or storage parameters

It is **not** intended for:
- Actual market price forecasting
- Investment planning or capacity assessment
- Policy analysis without re-calibrating generator parameters to official fleet data
- Multi-day or multi-zone studies without significant model extensions
