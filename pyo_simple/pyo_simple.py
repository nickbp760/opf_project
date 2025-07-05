import pyomo.environ as pyo

model = pyo.ConcreteModel()

# Variabel keputusan: jumlah meja dan kursi yang dibuat
model.x = pyo.Var(domain=pyo.NonNegativeReals)  # meja
model.y = pyo.Var(domain=pyo.NonNegativeReals)  # kursi

# Fungsi objektif: maksimalkan profit
model.obj = pyo.Objective(expr=40*model.x + 30*model.y, sense=pyo.maximize)

# Batasan:
model.constraint1 = pyo.Constraint(expr=2*model.x + 1*model.y <= 100)  # batas waktu kerja
model.constraint2 = pyo.Constraint(expr=3*model.x + 2*model.y <= 120)  # batas bahan

solver = pyo.SolverFactory('glpk')
solver.solve(model)

print(f"Jumlah meja: {model.x.value}")
print(f"Jumlah kursi: {model.y.value}")
