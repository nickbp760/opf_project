# simpan file ini sebagai scdc_mini_with_r.py
import pyomo.environ as pyo

print("1. Buat model Pyomo")
model = pyo.ConcreteModel(name="Mini_SCDC_with_R")

# --------------------------------------------------------------------------------------
print("2. Definisikan Set")
model.bus = pyo.Set(initialize=['Bus1', 'Bus2', 'Bus3'])
model.Gen = pyo.Set(initialize=['G1', 'G2'])
model.t = pyo.Set(initialize=[1, 2], ordered=True)
model.branch = pyo.Set(initialize=[('Bus1', 'Bus3'), ('Bus2', 'Bus3')], dimen=2)
model.GB = pyo.Set(initialize=[('Bus1', 'G1'), ('Bus2', 'G2')], dimen=2)
model.slack = pyo.Set(initialize=['Bus1'])
model.Sbase = pyo.Param(initialize=100)

# --------------------------------------------------------------------------------------
print("3. Definisikan Parameter")
gen_data = {
    'G1': {'a': 0.0, 'b': 10, 'c': 0, 'Pmin': 0, 'Pmax': 100, 'RU': 100, 'RD': 100},
    'G2': {'a': 0.0, 'b': 20, 'c': 0, 'Pmin': 0, 'Pmax': 100, 'RU': 100, 'RD': 100}
}
model.GD = pyo.Param(model.Gen, ['a', 'b', 'c', 'Pmin', 'Pmax', 'RU', 'RD'],
                     initialize=lambda m, g, p: gen_data[g][p])

bus_demand_raw = {
    ('Bus1', 1): 0, ('Bus2', 1): 0, ('Bus3', 1): 150,
    ('Bus1', 2): 0, ('Bus2', 2): 0, ('Bus3', 2): 100
}
model.BusData_pd = pyo.Param(model.bus, model.t, initialize=bus_demand_raw, default=0)

branch_data_raw = {
    ('Bus1', 'Bus3'): {'x': 0.1, 'r': 0.02, 'Limit': 100},
    ('Bus2', 'Bus3'): {'x': 0.1, 'r': 0.01, 'Limit': 120}
}
model.branch_x = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['x'])
model.branch_r = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['r'])
model.branch_Limit = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['Limit'])
model.bij = pyo.Param(model.branch, initialize=lambda m, i, j: 1 / m.branch_x[i, j])

# --------------------------------------------------------------------------------------
print("4. Definisikan Variable")
model.Pg = pyo.Var(model.Gen, model.t, domain=pyo.NonNegativeReals)
model.Pij = pyo.Var(model.branch, model.t, domain=pyo.Reals)
model.delta = pyo.Var(model.bus, model.t, domain=pyo.Reals, bounds=(-3.14, 3.14))
model.cost = pyo.Var(model.t, domain=pyo.NonNegativeReals)
model.OF = pyo.Var(domain=pyo.NonNegativeReals)
model.Ploss = pyo.Var(model.branch, model.t, domain=pyo.NonNegativeReals)

# --------------------------------------------------------------------------------------
print("5. Definisikan Constraint")

def power_flow_eq(m, i, j, t):
    return m.Pij[i, j, t] == m.bij[i, j] * (m.delta[i, t] - m.delta[j, t])
model.const1 = pyo.Constraint(model.branch, model.t, rule=power_flow_eq)

def power_balance(m, b, t):
    gen = sum(m.Pg[g, t] for g in m.Gen if (b, g) in m.GB)
    load = m.BusData_pd[b, t] / m.Sbase
    flow_out = sum(m.Pij[b, j, t] for j in m.bus if (b, j) in m.branch)
    flow_in = sum(m.Pij[j, b, t] for j in m.bus if (j, b) in m.branch)
    return gen - load == flow_out - flow_in
model.const2 = pyo.Constraint(model.bus, model.t, rule=power_balance)

model.pg_max = pyo.Constraint(model.Gen, model.t, rule=lambda m, g, t: m.Pg[g, t] <= m.GD[g, 'Pmax'] / m.Sbase)
model.pg_min = pyo.Constraint(model.Gen, model.t, rule=lambda m, g, t: m.Pg[g, t] >= m.GD[g, 'Pmin'] / m.Sbase)

for t in model.t:
    model.delta['Bus1', t].fix(0)

def pij_bounds(m, i, j, t):
    limit = m.branch_Limit[i, j] / m.Sbase
    return pyo.inequality(-limit, m.Pij[i, j, t], limit)
model.pij_bounds = pyo.Constraint(model.branch, model.t, rule=pij_bounds)

# Meskipun DCOPF mengabaikan r, untuk belajar kita bisa bikin model sederhana:
# Rugi daya (loss) di jalur 𝑖 → 𝑗 bisa dihitung dengan:
# P loss (i,j,t) = r(i,j) ​ ⋅ P (i,j,t) ^ 2
def power_loss_eq(m, i, j, t):
    return m.Ploss[i, j, t] == m.branch_r[i, j] * m.Pij[i, j, t]**2
model.power_loss = pyo.Constraint(model.branch, model.t, rule=power_loss_eq)

def cost_calc(m, t):
    return m.cost[t] == sum(m.GD[g, 'a'] * (m.Pg[g, t] * m.Sbase)**2 +
                            m.GD[g, 'b'] * (m.Pg[g, t] * m.Sbase) +
                            m.GD[g, 'c'] for g in m.Gen)
model.cost_thermal = pyo.Constraint(model.t, rule=cost_calc)

# Total cost includes the cost of generation and the cost of power loss, 5000 per MW of loss
model.total_cost = pyo.Constraint(expr=model.OF == sum(model.cost[t] for t in model.t) + 
                                  sum(model.Ploss[i, j, t] * 5000 for (i, j) in model.branch for t in model.t))

# --------------------------------------------------------------------------------------
print("6. Definisikan Objective Function")
model.objective = pyo.Objective(expr=model.OF, sense=pyo.minimize)

# --------------------------------------------------------------------------------------
print("7. Jalankan Solver")
# non linear solver
solver = pyo.SolverFactory('ipopt')
results = solver.solve(model, tee=False)

# --------------------------------------------------------------------------------------
print("\n=== Hasil Optimasi ===")
for t in model.t:
    print(f"\n=== Jam {t} ===")
    for g in model.Gen:
        print(f"{g}: {pyo.value(model.Pg[g, t]) * pyo.value(model.Sbase):.2f} MW")
    for (i, j) in model.branch:
        pij = pyo.value(model.Pij[i, j, t]) * model.Sbase.value
        loss = pyo.value(model.Ploss[i, j, t])
        print(f"Pij[{i}→{j}] = {pij:.2f} MW, Loss = {loss:.4f} MW")
    print(f"Biaya @t{t}: Rp {pyo.value(model.cost[t]):,.2f}")

print(f"\n>> Total Biaya + Biaya Rugi Daya: Rp {pyo.value(model.OF):,.2f}")
print(f">> Total Rugi Daya (semua waktu): {sum(pyo.value(model.Ploss[i,j,t]) for (i,j) in model.branch for t in model.t):.4f} MW")