# simpan file ini sebagai scdc_mini_upgraded.py
import pyomo.environ as pyo

print("1. Buat model Pyomo")
model = pyo.ConcreteModel(name="Mini_SCDC_Upgraded")

# --------------------------------------------------------------------------------------
print("2. Definisikan Set")
model.bus = pyo.Set(initialize=['Bus1', 'Bus2', 'Bus3'])
model.Gen = pyo.Set(initialize=['G1', 'G2'])
model.t = pyo.Set(initialize=[1, 2], ordered=True)
model.branch = pyo.Set(initialize=[('Bus1', 'Bus3'), ('Bus2', 'Bus3')], dimen=2)
model.GB = pyo.Set(initialize=[('Bus1', 'G1'), ('Bus2', 'G2')], dimen=2)
model.slack = pyo.Set(initialize=['Bus1'])
model.k = pyo.Set(initialize=[f'sg{i}' for i in range(1, 6)], ordered=True)  # 5 segmen

model.Sbase = pyo.Param(initialize=100)

# --------------------------------------------------------------------------------------
print("3. Definisikan Parameter")
gen_data = {
    'G1': {'a': 0.01, 'b': 10, 'c': 0, 'Pmin': 10, 'Pmax': 100, 'RU': 50, 'RD': 50},
    'G2': {'a': 0.02, 'b': 20, 'c': 0, 'Pmin': 20, 'Pmax': 100, 'RU': 40, 'RD': 40}
}
model.GD = pyo.Param(model.Gen, ['a', 'b', 'c', 'Pmin', 'Pmax', 'RU', 'RD'],
                     initialize=lambda m, g, p: gen_data[g][p])

bus_demand_raw = {
    ('Bus1', 1): 0, ('Bus2', 1): 0, ('Bus3', 1): 150,
    ('Bus1', 2): 0, ('Bus2', 2): 0, ('Bus3', 2): 100
}
model.BusData_pd = pyo.Param(model.bus, model.t, initialize=bus_demand_raw, default=0)

model.RE = pyo.Param(model.t, initialize={1: 1.0, 2: 0.8})

branch_data_raw = {
    ('Bus1', 'Bus3'): {'x': 0.1, 'Limit': 100},
    ('Bus2', 'Bus3'): {'x': 0.1, 'Limit': 120}
}
model.branch_x = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['x'])
model.branch_Limit = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['Limit'])
model.bij = pyo.Param(model.branch, initialize=lambda m, i, j: 1 / m.branch_x[i, j])

# Segmentasi Biaya
# Fungsi linier lebih simpel secara matematis Solver LP (linear programming)
# Bisa menyelesaikan ribuan variabel jauh lebih cepat dibanding solver NLP (non-linear programming)
# Tidak ada kurva bengkok — hanya “naik linear per segmen”
print("4. Hitung Piecewise Linear")
piecewise_data = {}
mincost_data = {}
for g in model.Gen:
    p_min = gen_data[g]['Pmin']
    p_max = gen_data[g]['Pmax']
    a = gen_data[g]['a']
    b = gen_data[g]['b']
    c = gen_data[g]['c']
    mincost_data[g] = a * p_min**2 + b * p_min + c
    dp = (p_max - p_min) / len(model.k)
    for idx, seg in enumerate(model.k):
        p_ini = p_min + idx * dp
        p_fin = p_ini + dp
        # cost min pada awal segmen tersebut
        c_ini = a * p_ini**2 + b * p_ini + c
        # cost max pada akhir segmen tersebut
        c_fin = a * p_fin**2 + b * p_fin + c
        slope = (c_fin - c_ini) / dp
        piecewise_data[g, seg] = {'s': slope, 'DP': dp}

model.Mincost = pyo.Param(model.Gen, initialize=mincost_data)
# Menyatakan: berapa biaya tambahan jika kamu menambah 1 MW di dalam segmen itu
# Artinya: biaya = slope × jumlah daya pada segmen itu slope×jumlah daya pada segmen itu
model.Slope = pyo.Param(model.Gen, model.k, initialize=lambda m, g, k: piecewise_data[g, k]['s'])
# Menyatakan: berapa lebar segmen itu, misalnya:
# segmen sg1: dari 10 MW sampai 28 MW → dp = 18
model.SegWidth = pyo.Param(model.Gen, model.k, initialize=lambda m, g, k: piecewise_data[g, k]['DP'])

# --------------------------------------------------------------------------------------
print("5. Definisikan Variable")
model.Pg = pyo.Var(model.Gen, model.t, domain=pyo.NonNegativeReals)
model.Pij = pyo.Var(model.branch, model.t, domain=pyo.Reals)
model.delta = pyo.Var(model.bus, model.t, domain=pyo.Reals, bounds=(-3.14, 3.14))
model.Pk = pyo.Var(model.Gen, model.t, model.k, domain=pyo.NonNegativeReals)
model.cost = pyo.Var(model.t, domain=pyo.NonNegativeReals)
model.OF = pyo.Var(domain=pyo.NonNegativeReals)

# --------------------------------------------------------------------------------------
print("6. Definisikan Constraint")
def power_flow_eq(m, i, j, t):
    return m.Pij[i, j, t] == m.bij[i, j] * (m.delta[i, t] - m.delta[j, t])
model.const1 = pyo.Constraint(model.branch, model.t, rule=power_flow_eq)

def power_balance(m, b, t):
    gen = sum(m.Pg[g, t] for g in m.Gen if (b, g) in m.GB)
    load = (m.RE[t] * m.BusData_pd[b, t]) / m.Sbase
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

# Di scdc_mini.py (versi awal), tidak ada constraint ramping sama sekali.
# Jadi generator bisa:
# Naik dari 0 ke 100 MW seketika
# Tanpa batas perubahan antar waktu
# dengan ramping daya harus naik/turun secara bertahap
# contoh:
# t = 2 (berdasarkan t=1 = 50 MW):
# Boleh naik maksimal 40 MW → maksimum: 50 + 40 = 90
# Boleh turun maksimal 30 MW → minimum: 50 - 30 = 20
# t = 3 (berdasarkan t=2 = 80 MW):
# Naik max 40 → 80 + 40 = 120 → tapi Pmax = 100
# Turun max 30 → 80 - 30 = 50
def ramp_up(m, g, t):
    if t == m.t.first(): return pyo.Constraint.Skip
    return m.Pg[g, t] - m.Pg[g, m.t.prev(t)] <= m.GD[g, 'RU'] / m.Sbase
model.ramp_up = pyo.Constraint(model.Gen, model.t, rule=ramp_up)

def ramp_down(m, g, t):
    if t == m.t.first(): return pyo.Constraint.Skip
    return m.Pg[g, m.t.prev(t)] - m.Pg[g, t] <= m.GD[g, 'RD'] / m.Sbase
model.ramp_down = pyo.Constraint(model.Gen, model.t, rule=ramp_down)

# aplikasi piecewise linear cost
def cost_piecewise(m, t):
    return m.cost[t] == sum(m.Mincost[g] + sum(m.Slope[g, k] * m.Pk[g, t, k] for k in m.k) for g in m.Gen)
model.cost_calc = pyo.Constraint(model.t, rule=cost_piecewise)

# Simulasi 2 Kombinasi Pk (cara Pyomo mengoptimasi)
# Kombinasi A (Normal): Pk[sg1] = 18 (maksimal)
# Pk[sg2] = 2 (sisa)
# Segmen	Rentang (MW)	Slope (Rp/MW)
# sg1	10–28	10.38
# sg2	28–46	10.74
# Total tambahan daya = 18 + 2 = 20 MW
# Biaya = 10.38 ⋅ 18 + 10.74 ⋅ 2 = 186.84 + 21.48 = 208.32
# Kombinasi B (Ngasal):
# Pk[sg1] = 10 Pk[sg2] = 10
# Biaya = 10.38 ⋅ 10 + 10.74 ⋅ 10 = 103.8 + 107.4 = 211.2
# lebih mahal karena tidak memanfaatkan segmen dengan biaya lebih murah

# constraint untuk memastikan total daya generator sesuai dengan segmen
def seg_sum(m, g, t):
    return m.Pg[g, t] * m.Sbase == m.GD[g, 'Pmin'] + sum(m.Pk[g, t, k] for k in m.k)
model.piecewise_sum = pyo.Constraint(model.Gen, model.t, rule=seg_sum)

# constraint untuk memastikan Pk tidak melebihi lebar segmen
def pk_bounds(m, g, t, k):
    return m.Pk[g, t, k] <= m.SegWidth[g, k]
model.pk_bound = pyo.Constraint(model.Gen, model.t, model.k, rule=pk_bounds)

model.total_cost = pyo.Constraint(expr=model.OF == sum(model.cost[t] for t in model.t))

# --------------------------------------------------------------------------------------
print("7. Definisikan Objective Function")
model.objective = pyo.Objective(expr=model.OF, sense=pyo.minimize)

print("8. Jalankan Solver")
solver = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=False)

print("\n=== Hasil Optimasi ===")
for t in model.t:
    print(f"\n=== Jam {t} ===")
    for g in model.Gen:
        print(f"{g}: {pyo.value(model.Pg[g, t]) * pyo.value(model.Sbase):.2f} MW")
    print(f"Biaya @t{t}: Rp {pyo.value(model.cost[t]):,.2f}")

print(f"\n>> Total Biaya: Rp {pyo.value(model.OF):,.2f}")
