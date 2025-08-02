import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
from lfybus_confirmed import build_ybus
from pyomo.environ import value
from lineflow_confirmed import lineflow
from busout_confirmed import busout
import sys  # noqa

# =============================
# Data Sistem
# =============================
print("0. Data System Initialized")
base_mva = 100  # Base MVA

bus_data = {
    1: {'type': 'Slack', 'Pd': 0,  'Qd': 0, 'V': 1.05, 'Vmin': 0.95, 'Vmax': 1.05},
    2: {'type': 'PV',    'Pd': 1,  'Qd': 2, 'V': 1.00, 'Vmin': 0.95, 'Vmax': 1.05},
    3: {'type': 'PQ',    'Pd': 20, 'Qd': 5, 'V': 1.00, 'Vmin': 0.95, 'Vmax': 1.05},
}

gen_bus_map = {
    1: 1,
    2: 2,
}

cost_data = {
    1: {'a': 0.01, 'b': 1.0, 'c': 5.0, 'Pmin': 0, 'Pmax': 8, 'Qmin': -50, 'Qmax': 50},
    2: {'a': 0.01, 'b': 1.00005, 'c': 10.0, 'Pmin': 0.005, 'Pmax': 30, 'Qmin': -50, 'Qmax': 50},
}

# From, To, R, X, B/2, Tap
linedata = np.array([
    [1, 2, 0.02, 0.06, 0.03, 1],
    [1, 3, 0.08, 0.24, 0.025, 1],
    [2, 3, 0.06, 0.18, 0.02, 1]
])
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
model.gen_bus = pyo.Param(model.GEN, initialize=gen_bus_map)

# Variabel
print("4. Definisikan Variable")
model.V = pyo.Var(model.BUS, initialize={i: bus_data[i]['V'] for i in bus_ids},
                  bounds=lambda m, i: (m.Vmin[i], m.Vmax[i]))
model.delta = pyo.Var(model.BUS, initialize=0.0)
model.Pg = pyo.Var(model.GEN, initialize={
    g: (cost_data[g]['Pmin'] + cost_data[g]['Pmax']) / (2) * base_mva for g in cost_data},
    bounds=lambda m, g: (m.Pmin[g], m.Pmax[g]))
model.Qg = pyo.Var(model.GEN, initialize={
    g: (cost_data[g]['Qmin'] + cost_data[g]['Qmax']) / (2) * base_mva for g in cost_data},
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

# Fungsi Objektif
print("6. Definisikan Objective Function")
def objective_rule(m):
    return sum(
        m.a[g] * m.Pg[g]**2 + m.b[g] * m.Pg[g] + m.c[g]
        for g in m.GEN
    )
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
else:
    print(f"⚠️ Solver selesai dengan status: {results.solver.status}, kondisi: {results.solver.termination_condition}")


print("8. Hasil Optimasi")
print("\n=== Output Generator ===")
total_cost = 0
total_qg = 0
for g in model.GEN:
    Pg = pyo.value(model.Pg[g]) * base_mva
    Qg = pyo.value(model.Qg[g]) * base_mva
    cost = cost_data[g]['a'] * Pg**2 + cost_data[g]['b'] * Pg + cost_data[g]['c']
    total_cost += cost
    total_qg += Qg
    print(f"Gen {g}: Pg = {Pg:.2f} MW, Qg = {Qg:.2f} Mvar, Cost = {cost:.2f}")

print(f"\nTotal Qg supplied (all gens): {total_qg:.2f} Mvar")
total_qd = sum(bus_data[i]['Qd'] for i in bus_ids)
print(f"Total Qd demand: {total_qd:.2f} Mvar")
print(f"Selisih Q (harus disuplai oleh jaringan / Slack): {total_qd - total_qg:.2f} Mvar")

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