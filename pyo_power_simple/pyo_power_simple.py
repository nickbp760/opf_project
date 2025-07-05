import pyomo.environ as pyo

# Buat model Pyomo
model = pyo.ConcreteModel()

# Variabel keputusan: daya dari generator G1 dan G2
model.P1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 100))  # G1
model.P2 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 100))  # G2

# Fungsi objektif: minimalkan biaya (G1 = Rp10/MW, G2 = Rp20/MW)
model.cost = pyo.Objective(expr=10 * model.P1 + 20 * model.P2, sense=pyo.minimize)

# Constraint: total daya harus 150 MW
model.demand_constraint = pyo.Constraint(expr=model.P1 + model.P2 == 150)

# Solver
solver = pyo.SolverFactory('glpk')  # atau 'ipopt' kalau glpk belum tersedia
result = solver.solve(model)

# Tampilkan hasil
print(f"Daya G1 (P1): {pyo.value(model.P1)} MW")
print(f"Daya G2 (P2): {pyo.value(model.P2)} MW")
print(f"Total Biaya: Rp {pyo.value(model.cost)}")