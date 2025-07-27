import pyomo.environ as pyo
import numpy as np

# =============================
# Data Sistem
# =============================

base_mva = 100  # Base MVA

bus_data = {
    1: {'type': 'Slack', 'Pd': 0,  'Qd': 0, 'V': 1.05, 'Vmin': 0.95, 'Vmax': 1.05},
    2: {'type': 'PV',    'Pd': 0,  'Qd': 1, 'V': 1.00, 'Vmin': 0.95, 'Vmax': 1.05},
    3: {'type': 'PQ',    'Pd': 20, 'Qd': 5, 'V': 1.00, 'Vmin': 0.95, 'Vmax': 1.05},
}

gen_bus_map = {
    1: 1,
    2: 2,
}

cost_data = {
    1: {'a': 0.01, 'b': 1.0, 'c': 5.0, 'Pmin': 0,      'Pmax': 100, 'Qmin': -50, 'Qmax': 50},
    2: {'a': 0.01, 'b': 1.00005, 'c': 10.0, 'Pmin': 0.005, 'Pmax': 100, 'Qmin': -50, 'Qmax': 50},
}

Ybus = np.array([
    [10-20j, -5+10j, -5+10j],
    [-5+10j, 10-20j, -5+10j],
    [-5+10j, -5+10j, 10-20j],
])

bus_ids = list(bus_data.keys())

# =============================
# Model Pyomo
# =============================

model = pyo.ConcreteModel()
model.BUS = pyo.Set(initialize=bus_ids)
model.GEN = pyo.Set(initialize=cost_data.keys())

# Variabel
model.V = pyo.Var(model.BUS, initialize={i: bus_data[i]['V'] for i in bus_ids},
                  bounds=lambda m, i: (bus_data[i]['Vmin'], bus_data[i]['Vmax']))

model.delta = pyo.Var(model.BUS, initialize=0.0)

model.Pg = pyo.Var(model.GEN, initialize=10, bounds=lambda m, g: (cost_data[g]['Pmin'], cost_data[g]['Pmax']))
model.Qg = pyo.Var(model.GEN, initialize=0, bounds=lambda m, g: (cost_data[g]['Qmin'], cost_data[g]['Qmax']))

# Constraint delta slack bus
def ref_bus_rule(m):
    return m.delta[1] == 0
model.ref_bus = pyo.Constraint(rule=ref_bus_rule)

# Power Balance P
def power_balance_P(m, i):
    Vi = m.V[i]
    deltai = m.delta[i]
    Pi_gen = sum(m.Pg[g] for g in model.GEN if gen_bus_map[g] == i)
    Pi_load = bus_data[i]['Pd'] / base_mva

    return Pi_gen - Pi_load == sum(
        Vi * m.V[j] * (
            Ybus[i-1, j-1].real * pyo.cos(deltai - m.delta[j]) +
            Ybus[i-1, j-1].imag * pyo.sin(deltai - m.delta[j])
        ) for j in bus_ids
    )
model.P_balance = pyo.Constraint(model.BUS, rule=power_balance_P)

# Power Balance Q
def power_balance_Q(m, i):
    if bus_data[i]['type'] in ['PQ', 'PV']:  # ✅ tambahkan PV
        Vi = m.V[i]
        deltai = m.delta[i]
        Qg = sum(m.Qg[g] for g in model.GEN if gen_bus_map[g] == i)
        Qd = bus_data[i]['Qd'] / base_mva
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
def objective_rule(m):
    return sum(
        cost_data[g]['a'] * m.Pg[g]**2 +
        cost_data[g]['b'] * m.Pg[g] +
        cost_data[g]['c']
        + 0.01 * m.Qg[g]**2  # ⬅️ Ini penalti untuk Qg (optional)
        for g in model.GEN
    )
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# =============================
# Solver
# =============================

solver = pyo.SolverFactory('ipopt')
results = solver.solve(model, tee=True)

# =============================
# Output
# =============================

print("\n=== HASIL AC-OPF ===")
for i in bus_ids:
    V = pyo.value(model.V[i])
    delta_deg = np.degrees(pyo.value(model.delta[i]))
    print(f"Bus {i}: V = {V:.4f} pu, δ = {delta_deg:.2f}°")

print("\n=== Output Generator ===")
total_cost = 0
for g in model.GEN:
    Pg = pyo.value(model.Pg[g]) * base_mva if model.Pg[g].value is not None else 0.0
    Qg = pyo.value(model.Qg[g]) * base_mva if model.Qg[g].value is not None else 0.0
    cost = (
        cost_data[g]['a'] * Pg**2 + cost_data[g]['b'] * Pg + cost_data[g]['c']
        + 0.01 * Qg**2  # ⬅️ Ini penalti untuk Qg (optional)
    )
    total_cost += cost
    print(f"Gen {g}: Pg = {Pg:.2f} MW, Qg = {Qg:.2f} Mvar, Cost = {cost:.2f}")

print(f"\nTotal Cost (pu basis): {pyo.value(model.obj):.4f}")
print(f"Total Cost (USD/jam atau sesuai satuan): {total_cost:.2f}")

total_Qg = sum(pyo.value(model.Qg[g]) for g in model.GEN) * base_mva
print(f"\nTotal Qg supplied: {total_Qg:.2f} Mvar (dibutuhkan: {bus_data[3]['Qd']:.2f} Mvar)")