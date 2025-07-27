import numpy as np
import pyomo.environ as pyo
from lfybus_confirmed import build_ybus
from lfnewton_confirmed import lfnewton
from lineflow_confirmed import lineflow
from busout_confirmed import busout

# -----------------------------
# 1. DATA SISTEM MINI
# -----------------------------
busdata = np.array([
    [1, 1, 1.05, 0,   0,   0,   10,  0,   0, 100, 0],  # Slack
    [2, 2, 1.00, 0,   0,   0,   10,  0,   0, 100, 0],  # PV
    [3, 0, 1.00, 0,  20,   5,    0,  0,   0,   0, 0],  # Load
])

linedata = np.array([
    [1, 2, 0.01, 0.02, 0, 1],
    [1, 3, 0.01, 0.03, 0, 1],
    [2, 3, 0.01, 0.025, 0, 1],
])

base_mva = 100

cost_data = {
    1: {'a': 0.01, 'b': 1.0, 'c': 0, 'Pmin': 0, 'Pmax': 100},
    2: {'a': 0.02, 'b': 1.0, 'c': 0, 'Pmin': 0, 'Pmax': 100},
}

# -----------------------------
# 2. ITERATIF OPF + LOAD FLOW
# -----------------------------
tolerance = 1e-4
max_iter = 10
max_lf_fail = 3  # max gagal loadflow
lf_fail_count = 0

# Inisialisasi pembangkitan rata
total_demand = np.sum(busdata[:, 4])
load_with_losses = total_demand
Pg_result = {}
SLT = None
print("=== ITERASI OPF + LOAD FLOW ===")
for it in range(max_iter):
    model = pyo.ConcreteModel()
    model.G = pyo.Set(initialize=[1, 2])
    model.Pg = pyo.Var(model.G, domain=pyo.NonNegativeReals)

    model.a = pyo.Param(model.G, initialize={g: cost_data[g]['a'] for g in model.G})
    model.b = pyo.Param(model.G, initialize={g: cost_data[g]['b'] for g in model.G})
    model.c = pyo.Param(model.G, initialize={g: cost_data[g]['c'] for g in model.G})
    model.Pmin = pyo.Param(model.G, initialize={g: cost_data[g]['Pmin'] for g in model.G})
    model.Pmax = pyo.Param(model.G, initialize={g: cost_data[g]['Pmax'] for g in model.G})

    def total_cost(model):
        return sum(model.a[g]*model.Pg[g]**2 + model.b[g]*model.Pg[g] for g in model.G)
    model.obj = pyo.Objective(rule=total_cost, sense=pyo.minimize)

    def power_balance(model):
        return sum(model.Pg[g] for g in model.G) == load_with_losses
    model.power_balance = pyo.Constraint(rule=power_balance)

    model.limit_min = pyo.Constraint(model.G, rule=lambda m, g: m.Pg[g] >= m.Pmin[g])
    model.limit_max = pyo.Constraint(model.G, rule=lambda m, g: m.Pg[g] <= m.Pmax[g])

    solver = pyo.SolverFactory('ipopt')
    result = solver.solve(model, tee=False)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"⚠️  Iterasi {it+1}: OPF tidak feasible.")
        break

    Pg1 = pyo.value(model.Pg[1])
    Pg2 = pyo.value(model.Pg[2])
    Pg_result = {1: Pg1, 2: Pg2}

    # Simpan busdata sementara
    temp_busdata = busdata.copy()
    temp_busdata[0, 6] = Pg1
    temp_busdata[1, 6] = Pg2

    Ybus = build_ybus(linedata, busdata.shape[0])
    Vm, delta_degree, Pg_new, Qg_new, Sbus = lfnewton(temp_busdata, Ybus, base_mva)

    if np.any(np.isnan(Vm)) or np.any(Vm < 0.95) or np.any(Vm > 1.05):
        print(f"❌ Iterasi {it+1}: Load flow gagal. Tegangan tidak dalam batas [0.95, 1.05] p.u.")
        lf_fail_count += 1
        if lf_fail_count >= max_lf_fail:
            print("⛔ Load flow gagal terus. Iterasi dihentikan.")
            break
        continue
    else:
        lf_fail_count = 0  # reset jika sukses
        # update nilai hasil OPF yang valid
        busdata[:, 6] = temp_busdata[:, 6]
        V = Vm * np.exp(1j * np.radians(delta_degree))
        results, SLT = lineflow(linedata, V, base_mva)
        new_load = total_demand + SLT.real

        print(f"✅ Iterasi {it+1}: Pg={Pg1+Pg2:.4f} MW, Load+Loss={new_load:.4f}, Loss={SLT.real:.4f}")

        if abs(new_load - load_with_losses) < tolerance:
            print(f"\n✅ Konvergen di iterasi ke-{it+1}")
            break
        else:
            load_with_losses = new_load

# -----------------------------
# 3. OUTPUT AKHIR
# -----------------------------
busout(busdata, Vm, delta_degree, Pg_new, Qg_new)

print("\nOptimal Generator Output (MW):")
for g in model.G:
    print(f"Generator {g}: {Pg_result[g]:.2f} MW")
print(f"\nTotal Cost: Rp {pyo.value(model.obj):,.2f}")

if SLT is not None:
    print(f"\nTotal System Losses: {SLT.real:.3f} MW, {SLT.imag:.3f} Mvar")

    print("\nLine Flow and Losses")
    print(f"{'From':>5} {'To':>5}   {'MW':>8} {'Mvar':>8} {'MVA':>8} {'MW_loss':>10} {'Mvar_loss':>10} {'Tap':>5}")
    for res in results:
        Snk = res['Snk']
        SL = res['SL']
        print(f"{res['from']:5d} {res['to']:5d}   {Snk.real:8.3f} {Snk.imag:8.3f} {abs(Snk):8.3f} {SL.real:10.3f} {SL.imag:10.3f} {res['tap']:5.2f}")
