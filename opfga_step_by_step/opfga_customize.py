import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
from lfybus_confirmed import build_ybus
from pyomo.environ import value
from lineflow_confirmed import lineflow
from busout_confirmed import busout
import pandas as pd
import numpy as np
import sys  # noqa

# =============================
# Data Sistem
# =============================
print("0. Data System Initialized")
base_mva = 100  # Base MVA
# Baca Excel
filename = 'opfga_step_by_step\system_data_costumize.xlsx'
bus_df = pd.read_excel(filename, sheet_name='bus_data')
gen_map_df = pd.read_excel(filename, sheet_name='gen_bus_map')
cost_df = pd.read_excel(filename, sheet_name='cost_data')
line_df = pd.read_excel(filename, sheet_name='linedata')

# Konversi ke dict: bus_data
bus_data = {}
for _, row in bus_df.iterrows():
    bus = int(row['bus'])
    bus_data[bus] = {
        'type': row['type'],
        'Pd': row['Pd'],
        'Qd': row['Qd'],
        'V': row['V'],
        'Vmin': row['Vmin'],
        'Vmax': row['Vmax'],
    }

# Konversi ke dict: gen_bus_map
# Mapping generator ke bus
gen_bus_map = dict(zip(gen_map_df['gen_id'], gen_map_df['bus_id']))
# List generator yang H2
h2_gens = gen_map_df.loc[gen_map_df['is_H2'] == 1, 'gen_id'].tolist()

# Konversi ke dict: cost_data
cost_data = {}
for _, row in cost_df.iterrows():
    gen_id = int(row['gen_id'])
    cost_data[gen_id] = {
        'a': row['a'],
        'b': row['b'],
        'c': row['c'],
        'Pmin': row['Pmin'],
        'Pmax': row['Pmax'],
        'Qmin': row['Qmin'],
        'Qmax': row['Qmax'],
        # Konsumsi H2 per MWh listrik
        # Lower Heating Value (LHV) H₂ = ± 33.33 kWh/kg
        # Contoh: jika efisiensi 50% → kg/MWh ≈ 1000/ (33.33 x 0.5) ≈ 60.0060006001 (isi sesuai datamu)
        'kH2': row.get('kH2', 0.0),
        # Kalau generator itu dispatch 10 MW selama 1 jam → 10 MWh output.
        # Konsumsi H₂ = 10 × 60 = 600 kg.
        # Biaya H₂ = 600 × 3 = 1800 $/h.
        # Harga H₂ konservatif 3 $/kg
        'priceH2': row.get('priceH2', 0.0),
    }

# Konversi ke np.array: linedata
linedata = line_df[['From', 'To', 'R', 'X', 'B_half', 'Tap', 'Smax']].to_numpy()

# Cek hasil
print("bus_data =", bus_data)
print("gen_bus_map =", gen_bus_map)
print("cost_data =", cost_data)
print("linedata =\n", linedata)

nbus = len(bus_data)
Ybus = build_ybus(linedata, nbus)

bus_ids = list(bus_data.keys())

# =============================
# Model Pyomo
# =============================
print("1. Buat model Pyomo")
model = pyo.ConcreteModel()

print("2. Definisikan Set")
model.BUS = pyo.Set(initialize=bus_ids)
model.GEN = pyo.Set(initialize=cost_data.keys())
model.H2GEN = pyo.Set(initialize=h2_gens)

# Parameter
print("3. Definisikan Parameter")
model.Vmin = pyo.Param(model.BUS, initialize={i: bus_data[i]['Vmin'] for i in bus_ids})
model.Vmax = pyo.Param(model.BUS, initialize={i: bus_data[i]['Vmax'] for i in bus_ids})
model.Pd = pyo.Param(model.BUS, initialize={i: bus_data[i]['Pd'] / base_mva for i in bus_ids})
model.Qd = pyo.Param(model.BUS, initialize={i: bus_data[i]['Qd'] / base_mva for i in bus_ids})
model.bus_type = pyo.Param(model.BUS, initialize={i: bus_data[i]['type'] for i in bus_ids})

model.Pmin = pyo.Param(model.GEN, initialize={g: cost_data[g]['Pmin'] / base_mva for g in cost_data})
model.Pmax = pyo.Param(model.GEN, initialize={g: cost_data[g]['Pmax'] / base_mva for g in cost_data})
model.Qmin = pyo.Param(model.GEN, initialize={g: cost_data[g]['Qmin'] / base_mva for g in cost_data})
model.Qmax = pyo.Param(model.GEN, initialize={g: cost_data[g]['Qmax'] / base_mva for g in cost_data})
model.a = pyo.Param(model.GEN, initialize={g: cost_data[g]['a'] for g in cost_data})
model.b = pyo.Param(model.GEN, initialize={g: cost_data[g]['b'] for g in cost_data})
model.c = pyo.Param(model.GEN, initialize={g: cost_data[g]['c'] for g in cost_data})
model.Smax = pyo.Param(range(1, len(linedata) + 1), initialize={
    i+1: linedata[i, 6] / base_mva for i in range(len(linedata))
})
model.gen_bus = pyo.Param(model.GEN, initialize=gen_bus_map)
model.kH2 = pyo.Param(model.GEN, initialize={g: cost_data[g]['kH2'] for g in cost_data})
model.priceH2 = pyo.Param(model.GEN, initialize={g: cost_data[g]['priceH2'] for g in cost_data})

# Variabel
print("4. Definisikan Variable")
model.V = pyo.Var(model.BUS, initialize={i: bus_data[i]['V'] for i in bus_ids},
                  bounds=lambda m, i: (m.Vmin[i], m.Vmax[i]))
model.delta = pyo.Var(model.BUS, initialize=0.0)
model.Pg = pyo.Var(model.GEN, 
    initialize={g: (cost_data[g]['Pmin'] + cost_data[g]['Pmax']) / (2 * base_mva) for g in cost_data},
    bounds=lambda m, g: (m.Pmin[g], m.Pmax[g]))

model.Qg = pyo.Var(model.GEN, 
    initialize={g: (cost_data[g]['Qmin'] + cost_data[g]['Qmax']) / (2 * base_mva) for g in cost_data},
    bounds=lambda m, g: (m.Qmin[g], m.Qmax[g]))

print("5. Definisikan Constraint")
def ref_bus_rule(m):
    return m.delta[1] == 0
model.ref_bus = pyo.Constraint(rule=ref_bus_rule)

# Power balance P
def power_balance_P(m, i):
    Vi = m.V[i]
    deltai = m.delta[i]
    Pi_gen = sum(m.Pg[g] for g in m.GEN if m.gen_bus[g] == i)
    Pi_load = m.Pd[i]

    return Pi_gen - Pi_load == sum(
        Vi * m.V[j] * (
            Ybus[i-1, j-1].real * pyo.cos(deltai - m.delta[j]) +
            Ybus[i-1, j-1].imag * pyo.sin(deltai - m.delta[j])
        ) for j in bus_ids
    )
model.P_balance = pyo.Constraint(model.BUS, rule=power_balance_P)

# Power balance Q
def power_balance_Q(m, i):
    if m.bus_type[i] in ['PQ', 'PV', 'Slack']:
        Vi = m.V[i]
        deltai = m.delta[i]
        Qg = sum(m.Qg[g] for g in m.GEN if m.gen_bus[g] == i)
        Qd = m.Qd[i]
        return Qg - Qd == sum(
            Vi * m.V[j] * (
                Ybus[i-1, j-1].real * pyo.sin(deltai - m.delta[j]) -
                Ybus[i-1, j-1].imag * pyo.cos(deltai - m.delta[j])
            ) for j in bus_ids
        )
    else:
        return pyo.Constraint.Skip
model.Q_balance = pyo.Constraint(model.BUS, rule=power_balance_Q)

model.Sij_limit = pyo.ConstraintList()
use_smax_constraint = True
if use_smax_constraint:
    for k in range(len(linedata)):
        i = int(linedata[k, 0])
        j = int(linedata[k, 1])
        G = Ybus[i-1, j-1].real
        B = Ybus[i-1, j-1].imag
        Smax_sq = model.Smax[k+1] ** 2

        Vi = model.V[i]
        Vj = model.V[j]
        di = model.delta[i]
        dj = model.delta[j]

        # Pij dan Qij berdasarkan rumus AC power flow
        Pij = Vi**2 * G - Vi * Vj * (G * pyo.cos(di - dj) + B * pyo.sin(di - dj))
        Qij = -Vi**2 * B - Vi * Vj * (G * pyo.sin(di - dj) - B * pyo.cos(di - dj))

        model.Sij_limit.add(Pij**2 + Qij**2 <= Smax_sq)

# Fungsi Objektif
print("6. Definisikan Objective Function")
def objective_rule(m):
    # Gen cost: a*(Pg_MW)^2 + b*(Pg_MW) + c
    gen_cost = sum(
        m.a[g] * (m.Pg[g] * base_mva)**2 +
        m.b[g] * (m.Pg[g] * base_mva) +
        m.c[g]
        for g in m.GEN
    )

    # Biaya H2 untuk H2GEN: priceH2 ($/kg) * kH2 (kg/MWh) * Pg_MW
    h2_fuel_cost = sum(
        m.priceH2[g] * m.kH2[g] * (m.Pg[g] * base_mva)
        for g in m.H2GEN
    )

    return gen_cost + h2_fuel_cost

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# =============================
# Solver
# =============================
print("7. Jalankan Optimasi")
solver = pyo.SolverFactory('ipopt')
results = solver.solve(model, tee=True)

print("\n=== Hasil Solusi ===")
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("✅ Solusi optimal ditemukan!")
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("❌ Tidak ada solusi yang memenuhi constraint (infeasible).")
    sys.exit(0)
else:
    print(f"⚠️ Solver selesai dengan status: {results.solver.status}, kondisi: {results.solver.termination_condition}")
    sys.exit(0)


print("8. Hasil Optimasi")
print("\n=== Output Generator ===")
total_cost = 0.0
total_h2_cost = 0.0
total_gen_cost = 0.0
for g in model.GEN:
    Pg_MW = pyo.value(model.Pg[g]) * base_mva
    Qg = pyo.value(model.Qg[g]) * base_mva

    # Biaya produksi (MW basis) — konsisten dengan objektif yang baru
    cost_gen = cost_data[g]['a'] * (Pg_MW**2) + cost_data[g]['b'] * Pg_MW + cost_data[g]['c']

    # Biaya H2 (jika generator termasuk H2GEN)
    if g in list(model.H2GEN):
        h2_cost = pyo.value(model.priceH2[g]) * pyo.value(model.kH2[g]) * Pg_MW
    else:
        h2_cost = 0.0

    total_cost += cost_gen + h2_cost
    total_h2_cost += h2_cost
    total_gen_cost += cost_gen

    fuel_tag = " (H2)" if g in list(model.H2GEN) else ""
    print(f"Gen {g}{fuel_tag}: Pg = {Pg_MW:.2f} MW, Qg = {Qg:.2f} Mvar, "
          f"Cost_gen = {cost_gen:.2f}, H2_cost = {h2_cost:.2f}, Total = {cost_gen + h2_cost:.2f}")

print(f"\nTotal Cost (gen + H2): {total_cost:.2f}")
print(f"Total H2 Gen cost: {total_h2_cost:.2f}")
print(f"Total Normal Gen cost: {total_gen_cost:.2f}")

print("9. Hasil Bus Out")
Vm = np.array([value(model.V[i]) for i in model.BUS])
delta_rad = np.array([value(model.delta[i]) for i in model.BUS])
delta_deg = np.degrees(delta_rad)

Pg_arr = np.zeros(nbus)
Qg_arr = np.zeros(nbus)
for g in model.GEN:
    b = model.gen_bus[g]
    Pg_arr[b - 1] += value(model.Pg[g]) * base_mva
    Qg_arr[b - 1] += value(model.Qg[g]) * base_mva

busdata_np = np.array([
    [
        i,
        1 if bus_data[i]['type'] == 'Slack' else 2 if bus_data[i]['type'] == 'PV' else 0,
        bus_data[i]['V'],             # Vinit
        0,                            # θinit
        bus_data[i]['Pd'],
        bus_data[i]['Qd'],
        0, 0,                        # Pg, Qg (diisi Pg_arr, Qg_arr)
        bus_data[i]['Vmin'],         # Vmin
        bus_data[i]['Vmax'],         # Vmax
        0                            # Qsh (diabaikan)
    ]
    for i in bus_data
])

busout(busdata_np, Vm, delta_deg, Pg_arr, Qg_arr)

print("\n9. Hasil Line Flow")
V_complex = Vm * np.exp(1j * delta_rad)
line_results, SLT = lineflow(linedata, V_complex, base_mva)

print("\n=== Hasil Line Flow ===")
for res in line_results:
    f, t = res['from'], res['to']
    Snk, Skn, SL = res['Snk'], res['Skn'], res['SL']
    print(f"Line {f} → {t}:")
    print(f"  S{f}{t} = {Snk.real:.4f} + j{Snk.imag:.4f} MVA")
    print(f"  S{t}{f} = {Skn.real:.4f} + j{Skn.imag:.4f} MVA")
    print(f"  Loss    = {SL.real:.4f} + j{SL.imag:.4f} MVA\n")

print(f"Total system losses: {SLT.real:.4f} + j{SLT.imag:.4f} MVA")

print("\n🔎 Cek Magnitude Line Flow:")
for res in line_results:
    Sij_mag = abs(res['Snk'])
    print(f"Line {res['from']} → {res['to']}: |Sij| = {Sij_mag:.4f} MVA")