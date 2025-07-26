import numpy as np
import pyomo.environ as pyo
from lfybus import build_ybus
from lfnewton import lfnewton
from lineflow import lineflow
from busout import busout

# -----------------------------
# 1. DATA INPUT
# -----------------------------
#  =============================
# BUS DATA
# =============================
# Format kolom:
# [Bus ID, Type, Vm, delta, Pd, Qd, Pg, Qg, Qmin, Qmax, Qsh]
# Type: 0 = PQ bus, 1 = Slack bus, 2 = PV bus

busdata = np.array([
    # bus, type, Vm, delta, Pd, Qd, Pg, Qg, Qmin, Qmax, Qsh
    [1, 1, 1.06, 0,     0,    0,   232.4, -16.9, 0, 0, 0],
    [2, 2, 1.045, 0,   21.7, 12.7, 40,    42.4,  0, 0, 0],
    [3, 0, 1.01,  0,   94.2, 19.0, 0,     0,     0, 0, 0],
])

# =============================
# LINE DATA
# =============================
# Format kolom:
# [From Bus, To Bus, R (p.u.), X (p.u.), Bc (p.u.), a (tap ratio)]
# - R  = resistance (hambatan) dari saluran
# - X  = reactance (reaktansi induktif)
# - Bc = susceptance (kapasitansi shunt per saluran)
# - a  = tap ratio dari trafo (1 artinya tidak ada tap changer)

linedata = np.array([
    # from, to,   R (p.u.), X (p.u.),  B/2 (p.u.), tap
    [1,    2,     0.02,     0.06,      0.03,       1],   # Saluran antara Bus 1 dan Bus 2
    [1,    3,     0.08,     0.24,      0.025,      1],   # Saluran antara Bus 1 dan Bus 3
    [2,    3,     0.06,     0.18,      0.02,       1],   # Saluran antara Bus 2 dan Bus 3
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

# 7. UPDATE busdata & RUN FLOW
# -----------------------------
Pg1 = pyo.value(model.Pg[1])
Pg2 = pyo.value(model.Pg[2])

# Perbaikan disini! Jangan masukkan Pg2 ke bus 3 yang bukan generator
busdata[0, 6] = Pg1  # Bus 1 (Slack, Generator 1)
busdata[1, 6] = Pg2  # Bus 2 (PV, Generator 2)

Ybus = build_ybus(linedata, busdata.shape[0])
Vm, delta_deg, Pg, Qg, S, P_calc, Q_calc = lfnewton(busdata, Ybus, base_mva)
busout(busdata, Vm, delta_deg, Pg, Qg, P_calc, Q_calc)

# -----------------------------
# 8. PRINT HASIL
# -----------------------------
print("\nOptimal Generator Output (MW):")
for g in model.G:
    print(f"Generator {g}: {pyo.value(model.Pg[g]):.2f} MW")

print(f"\nTotal Cost: Rp {pyo.value(model.obj):,.2f}")

# ----------------------------
# 9. HITUNG LINE FLOW
# ----------------------------
V = Vm * np.exp(1j * np.radians(delta_deg))
results, SLT = lineflow(linedata, V, base_mva)

print("\nLine Flow and Losses")
print(f"{'From':>5} {'To':>5}   {'MW':>8} {'Mvar':>8} {'MVA':>8} {'MW_loss':>10} {'Mvar_loss':>10} {'Tap':>5}")
for res in results:
    Snk = res['Snk']
    SL = res['SL']
    print(f"{res['from']:5d} {res['to']:5d}   {Snk.real:8.3f} {Snk.imag:8.3f} {abs(Snk):8.3f} {SL.real:10.3f} {SL.imag:10.3f} {res['tap']:5.2f}")

print(f"\nTotal line loss: {SLT.real:.3f} MW, {SLT.imag:.3f} Mvar")
