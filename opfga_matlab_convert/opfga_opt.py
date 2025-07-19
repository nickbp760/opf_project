import numpy as np
import pyomo.environ as pyo
from lfybus import build_ybus
from lfnewton import lfnewton

# -----------------------------
# 1. DATA INPUT
# -----------------------------
busdata = np.array([
    [1, 1, 1.06, 0,     0,    0,   0,    0, 0, 0, 0],
    [2, 2, 1.045, 0,   21.7, 12.7, 0,    0, 0, 0, 0],
    [3, 0, 1.01,  0,   94.2, 19.0, 0,    0, 0, 0, 0],
])

linedata = np.array([
    [1, 2, 0.02, 0.06, 0.03, 1],
    [1, 3, 0.08, 0.24, 0.025, 1],
    [2, 3, 0.06, 0.18, 0.02, 1],
])

base_mva = 100

cost_data = {
    1: {'a': 0.02, 'b': 2, 'c': 0, 'Pmin': 10, 'Pmax': 80},
    2: {'a': 0.0175, 'b': 1.75, 'c': 0, 'Pmin': 2, 'Pmax': 80},
}

# -----------------------------
# 2. SETUP PYOMO MODEL
# -----------------------------
model = pyo.ConcreteModel()
model.G = pyo.Set(initialize=[1, 2])
model.Pg = pyo.Var(model.G, domain=pyo.NonNegativeReals)

model.a = pyo.Param(model.G, initialize={g: cost_data[g]['a'] for g in cost_data})
model.b = pyo.Param(model.G, initialize={g: cost_data[g]['b'] for g in cost_data})
model.c = pyo.Param(model.G, initialize={g: cost_data[g]['c'] for g in cost_data})
model.Pmin = pyo.Param(model.G, initialize={g: cost_data[g]['Pmin'] for g in cost_data})
model.Pmax = pyo.Param(model.G, initialize={g: cost_data[g]['Pmax'] for g in cost_data})

# -----------------------------
# 3. OBJECTIVE FUNCTION
# -----------------------------
def total_cost(model):
    return sum(model.a[g]*model.Pg[g]**2 + model.b[g]*model.Pg[g] + model.c[g] for g in model.G)
model.obj = pyo.Objective(rule=total_cost, sense=pyo.minimize)

# -----------------------------
# 4. CONSTRAINT: TOTAL LOAD
# -----------------------------
total_load = np.sum(busdata[:, 4])

def power_balance(model):
    return sum(model.Pg[g] for g in model.G) == total_load
model.power_balance = pyo.Constraint(rule=power_balance)

# -----------------------------
# 5. CONSTRAINT: LIMITS
# -----------------------------
model.limit_min = pyo.Constraint(model.G, rule=lambda m, g: m.Pg[g] >= m.Pmin[g])
model.limit_max = pyo.Constraint(model.G, rule=lambda m, g: m.Pg[g] <= m.Pmax[g])

# -----------------------------
# 6. SOLVE MODEL
# -----------------------------
solver = pyo.SolverFactory('ipopt')
results = solver.solve(model)

# -----------------------------
# 7. UPDATE busdata & RUN FLOW
# -----------------------------
Pg1 = pyo.value(model.Pg[1])
Pg2 = pyo.value(model.Pg[2])

busdata[1, 6] = Pg1  # bus 2, generator
busdata[2, 6] = Pg2  # bus 3, generator

Ybus = build_ybus(linedata, busdata.shape[0])
Vm, delta_deg, Pg_out, Qg, S, P_calc, Q_calc = lfnewton(busdata, Ybus, base_mva)

# -----------------------------
# 8. PRINT HASIL
# -----------------------------
print("\nOptimal Generator Output (MW):")
for g in model.G:
    print(f"Generator {g}: {pyo.value(model.Pg[g]):.2f} MW")

print(f"\nTotal Cost: Rp {pyo.value(model.obj):,.2f}")