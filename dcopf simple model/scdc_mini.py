# simpan file ini sebagai scdc_mini.py
import pyomo.environ as pyo

model = pyo.ConcreteModel(name="Mini_SCDC")

# set digunakan untuk mendefinisikan himpunan
model.bus = pyo.Set(initialize=['Bus1', 'Bus2', 'Bus3'])
model.Gen = pyo.Set(initialize=['G1', 'G2'])
model.t = pyo.Set(initialize=[1, 2], ordered=True)
model.branch = pyo.Set(initialize=[('Bus1', 'Bus3'), ('Bus2', 'Bus3')], dimen=2)
model.GB = pyo.Set(initialize=[('Bus1', 'G1'), ('Bus2', 'G2')], dimen=2)
# slack adalah acuan untuk bus
# Pij =bij⋅(δi​−δj​)
model.slack = pyo.Set(initialize=['Bus1'])

model.Sbase = pyo.Param(initialize=100)

# param adalah nilai pada himpunan
# misalnya set generator memiliki nilai parameter seperti a, b, c, Pmin, Pmax, RU, RD
gen_data = {
    'G1': {'a': 0.0, 'b': 10, 'c': 0, 'Pmin': 0, 'Pmax': 100, 'RU': 100, 'RD': 100},
    'G2': {'a': 0.0, 'b': 20, 'c': 0, 'Pmin': 0, 'Pmax': 100, 'RU': 100, 'RD': 100}
}
model.GD = pyo.Param(model.Gen, ['a', 'b', 'c', 'Pmin', 'Pmax', 'RU', 'RD'],
                     initialize=lambda m, g, p: gen_data[g][p])
print("Generator Data (GD):", model.GD)
# lambda untuk mengakses data generator, m adalah model yang merupakan format pyomo
# m.GD[g, 'a'] mengakses parameter 'a' dari generator g
# contoh Biaya(P)=a⋅P^2+b⋅P+c
# Pmin = 0 → tidak ada daya minimum
# Pmax = 100 → tidak boleh menghasilkan lebih dari 100 MW
# P[t=1] = 50 MW
# RU = 20 MW
# RD = 10 MW
# 40 <= p[t=2] <= 70
#Daya yang dihasilkan oleh pembangkit bisa naik atau turun karena:
# 🔹 1. Perubahan Beban (Load)
# Saat konsumsi listrik naik (misal siang hari), daya harus naik.
# Saat malam atau rendah pemakaian, daya bisa turun.
# 🔹 2. Perubahan kondisi sistem
# Tambahan pembangkitan dari renewable (matahari, angin)
# Gangguan sistem (trip, overload)

bus_demand_raw = {
    ('Bus1', 1): 0, ('Bus2', 1): 0, ('Bus3', 1): 150,
    ('Bus1', 2): 0, ('Bus2', 2): 0, ('Bus3', 2): 100
}
# Ini adalah dictionary Python yang menyatakan:
# Berapa banyak beban listrik (dalam MW) yang dibutuhkan di setiap bus untuk setiap waktu.
model.BusData_pd = pyo.Param(model.bus, model.t, initialize=bus_demand_raw, default=0)
print("BusData_pd:", model.BusData_pd)

branch_data_raw = {
    ('Bus1', 'Bus3'): {'x': 0.1, 'Limit': 100},
    ('Bus2', 'Bus3'): {'x': 0.1, 'Limit': 120}
}
# Pij = 1/xij⋅(δi​−δj)
# 0.1 berarti xij adalah 10
# xij : reaktansi jalur dari bus i ke bus j (hambatan)
# Jalur dari Bus1 ke Bus3 hanya boleh dilewati maksimal 100 MW, baik ke arah Bus3 atau sebaliknya (tergantung tanda delta).

model.branch_x = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['x'])
model.branch_Limit = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['Limit'])
model.bij = pyo.Param(model.branch, initialize=lambda m, i, j: 1 / m.branch_x[i, j])

model.Pg = pyo.Var(model.Gen, model.t, domain=pyo.NonNegativeReals)
model.Pij = pyo.Var(model.branch, model.t, domain=pyo.Reals)
model.delta = pyo.Var(model.bus, model.t, domain=pyo.Reals, bounds=(-3.14, 3.14))
model.cost = pyo.Var(model.t, domain=pyo.NonNegativeReals)
model.OF = pyo.Var(domain=pyo.NonNegativeReals)

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

def cost_calc(m, t):
    return m.cost[t] == sum(m.GD[g, 'a'] * (m.Pg[g, t] * m.Sbase)**2 +
                            m.GD[g, 'b'] * (m.Pg[g, t] * m.Sbase) +
                            m.GD[g, 'c'] for g in m.Gen)
model.cost_thermal = pyo.Constraint(model.t, rule=cost_calc)

model.total_cost = pyo.Constraint(expr=model.OF == sum(model.cost[t] for t in model.t))
model.objective = pyo.Objective(expr=model.OF, sense=pyo.minimize)

# ======= Jalankan Solver =========
solver = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Tampilkan hasil
for t in model.t:
    print(f"\n=== Jam {t} ===")
    for g in model.Gen:
        # print(f"{g}: {pyo.value(model.Pg[g, t]) * model.Sbase:.2f} MW")
        print(f"{g}: {pyo.value(model.Pg[g, t]) * pyo.value(model.Sbase):.2f} MW")
    print(f"Total Cost @t{t}: Rp {pyo.value(model.cost[t]):,.2f}")

print(f"\n>> Total Biaya Seluruhnya: Rp {pyo.value(model.OF):,.2f}")
