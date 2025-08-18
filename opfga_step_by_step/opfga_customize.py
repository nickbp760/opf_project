import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
from lfybus_confirmed import build_ybus
from pyomo.environ import value
from lineflow_confirmed import lineflow
from busout_confirmed import busout
import pandas as pd
import sys

print("0. Data System Initialized")
base_mva = 100
filename = r"opfga_step_by_step\system_data_costumize.xlsx"

# === Read sheets ===
bus_df     = pd.read_excel(filename, sheet_name='bus_data')
gen_map_df = pd.read_excel(filename, sheet_name='gen_bus_map')
thermal_df = pd.read_excel(filename, sheet_name='gen_thermal')   # a,b,c only for thermal gens
h2_df      = pd.read_excel(filename, sheet_name='gen_hidro')     # H2 params for H2 gens
line_df    = pd.read_excel(filename, sheet_name='linedata')

# === Build dictionaries ===
# bus dict
bus_data = {}
for _, row in bus_df.iterrows():
    i = int(row['bus'])
    bus_data[i] = {
        'type': row['type'],
        'Pd'  : float(row['Pd']),
        'Qd'  : float(row['Qd']),
        'V'   : float(row['V']),
        'Vmin': float(row['Vmin']),
        'Vmax': float(row['Vmax']),
    }

# generator -> bus and limits (from gen_bus_map)
gen_bus_map = {}
limits_data = {}
h2_gens = []
for _, r in gen_map_df.iterrows():
    g = int(r['gen_id'])
    gen_bus_map[g] = int(r['bus_id'])
    limits_data[g] = {
        'Pmin': float(r['Pmin']),
        'Pmax': float(r['Pmax']),
        'Qmin': float(r['Qmin']),
        'Qmax': float(r['Qmax']),
    }
    if int(r['is_H2']) == 1:
        h2_gens.append(g)

# thermal cost dict (a,b,c) — only rows present in sheet
thermal_cost = {int(r['gen_id']): {'a': float(r['a']), 'b': float(r['b']), 'c': float(r['c'])}
                for _, r in thermal_df.iterrows()}

# H2 parameters dict — only rows present in sheet
h2_param = {int(r['gen_id']): {'kH2': float(r['kH2']),
                               'priceH2': float(r['priceH2']),
                               'H2_cap': float(r['H2_cap']),
                               'H2_init': float(r['H2_init'])}
            for _, r in h2_df.iterrows()}

# lines
linedata = line_df[['From','To','R','X','B_half','Tap','Smax']].to_numpy(dtype=float)

ramp_up = {}
ramp_down = {}
Pg_init = {}
for _, r in gen_map_df.iterrows():
    g = int(r['gen_id'])
    ramp_up[g] = float(r['Ramp_up'])/base_mva if 'Ramp_up' in r and not pd.isna(r['Ramp_up']) else 9999/base_mva
    ramp_down[g] = float(r['Ramp_down'])/base_mva if 'Ramp_down' in r and not pd.isna(r['Ramp_down']) else 9999/base_mva
    Pg_init[g] = float(r['Pg_init'])/base_mva if 'Pg_init' in r and not pd.isna(r['Pg_init']) else limits_data[g]['Pmin']/base_mva

print("bus_data =", bus_data)
print("gen_bus_map =", gen_bus_map)
print("h2_gens =", h2_gens)
print("thermal_cost =", thermal_cost)
print("h2_param =", h2_param)
print("linedata =\n", linedata)

nbus = len(bus_data)
Ybus = build_ybus(linedata, nbus)
bus_ids = sorted(bus_data.keys())
assert bus_ids == list(range(1, len(bus_ids)+1)), "Bus ID harus 1..nbus"
gen_ids = list(gen_bus_map.keys())

# === Pyomo model ===
print("1. Buat model Pyomo")
m = pyo.ConcreteModel()

print("2. Definisikan Set")
m.BUS = pyo.Set(initialize=bus_ids)
m.GEN = pyo.Set(initialize=gen_ids)
m.H2GEN = pyo.Set(initialize=h2_gens)
m.TIME = pyo.Set(initialize=[1, 2])

print("3. Definisikan Parameter")
m.Vmin = pyo.Param(m.BUS, initialize={i: bus_data[i]['Vmin'] for i in bus_ids})
m.Vmax = pyo.Param(m.BUS, initialize={i: bus_data[i]['Vmax'] for i in bus_ids})
m.Pd   = pyo.Param(m.BUS, initialize={i: bus_data[i]['Pd']/base_mva for i in bus_ids})
m.Qd   = pyo.Param(m.BUS, initialize={i: bus_data[i]['Qd']/base_mva for i in bus_ids})
m.bus_type = pyo.Param(m.BUS, initialize={i: bus_data[i]['type'] for i in bus_ids}, within=pyo.Any)

# limits
m.Pmin = pyo.Param(m.GEN, initialize={g: limits_data[g]['Pmin']/base_mva for g in gen_ids})
m.Pmax = pyo.Param(m.GEN, initialize={g: limits_data[g]['Pmax']/base_mva for g in gen_ids})
m.Qmin = pyo.Param(m.GEN, initialize={g: limits_data[g]['Qmin']/base_mva for g in gen_ids})
m.Qmax = pyo.Param(m.GEN, initialize={g: limits_data[g]['Qmax']/base_mva for g in gen_ids})
m.RampUp = pyo.Param(m.GEN, initialize=ramp_up)
m.RampDown = pyo.Param(m.GEN, initialize=ramp_down)
m.Pg_init = pyo.Param(m.GEN, initialize=Pg_init)

# thermal cost (default 0 for non-thermal)
m.a = pyo.Param(m.GEN, initialize={g: thermal_cost.get(g, {}).get('a', 0.0) for g in gen_ids})
m.b = pyo.Param(m.GEN, initialize={g: thermal_cost.get(g, {}).get('b', 0.0) for g in gen_ids})
m.c = pyo.Param(m.GEN, initialize={g: thermal_cost.get(g, {}).get('c', 0.0) for g in gen_ids})

# line Smax
m.Smax = pyo.Param(range(1, len(linedata)+1),
                   initialize={i+1: linedata[i,6]/base_mva for i in range(len(linedata))})

# map gen->bus
m.gen_bus = pyo.Param(m.GEN, initialize=gen_bus_map)

# H2 params (default 0 for non-H2)
m.kH2     = pyo.Param(m.GEN, initialize={g: h2_param.get(g, {}).get('kH2', 0.0) for g in gen_ids})
m.priceH2 = pyo.Param(m.GEN, initialize={g: h2_param.get(g, {}).get('priceH2', 0.0) for g in gen_ids})
m.H2_cap0 = pyo.Param(m.GEN, initialize={g: h2_param.get(g, {}).get('H2_cap', 0.0) for g in gen_ids})
m.H2_init0= pyo.Param(m.GEN, initialize={g: h2_param.get(g, {}).get('H2_init', 0.0) for g in gen_ids})

# === Variables ===
print("4. Definisikan Variable")
m.V = pyo.Var(m.BUS, m.TIME,
              initialize={(i,t): bus_data[i]['V'] for i in bus_ids for t in m.TIME},
              bounds=lambda m,i,t: (m.Vmin[i], m.Vmax[i]))

m.delta = pyo.Var(m.BUS, m.TIME, initialize=0.0)
m.Pg = pyo.Var(m.GEN, m.TIME,
    initialize={(g,t): (limits_data[g]['Pmin']+limits_data[g]['Pmax'])/(2*base_mva)
                 for g in gen_ids for t in m.TIME},
    bounds=lambda m,g,t: (m.Pmin[g], m.Pmax[g]))

m.Qg = pyo.Var(m.GEN, m.TIME,
    initialize={(g,t): (limits_data[g]['Qmin']+limits_data[g]['Qmax'])/(2*base_mva)
                 for g in gen_ids for t in m.TIME},
    bounds=lambda m,g,t: (m.Qmin[g], m.Qmax[g]))

# H2 inventory & purchase only for H2 gens
m.H2_level = pyo.Var(m.H2GEN, m.TIME, within=pyo.NonNegativeReals)
m.H2_buy   = pyo.Var(m.H2GEN, m.TIME, within=pyo.NonNegativeReals)

print("5. Definisikan Constraint")
# Reference bus angle for each time
m.ref_bus = pyo.Constraint(m.TIME, rule=lambda m,t: m.delta[1,t] == 0)

def power_balance_P(m, i, t):
    Vi = m.V[i,t]; di = m.delta[i,t]
    Pgen = sum(m.Pg[g,t] for g in m.GEN if m.gen_bus[g]==i)
    return Pgen - m.Pd[i] == sum(
        Vi*m.V[j,t]*(Ybus[i-1,j-1].real*pyo.cos(di - m.delta[j,t]) +
                     Ybus[i-1,j-1].imag*pyo.sin(di - m.delta[j,t]))
        for j in bus_ids
    )
m.P_balance = pyo.Constraint(m.BUS, m.TIME, rule=power_balance_P)

def power_balance_Q(m, i, t):
    if m.bus_type[i] in ['PQ','PV','Slack']:
        Vi = m.V[i,t]; di = m.delta[i,t]
        Qgen = sum(m.Qg[g,t] for g in m.GEN if m.gen_bus[g]==i)
        return Qgen - m.Qd[i] == sum(
            Vi*m.V[j,t]*(Ybus[i-1,j-1].real*pyo.sin(di - m.delta[j,t]) -
                         Ybus[i-1,j-1].imag*pyo.cos(di - m.delta[j,t]))
            for j in bus_ids
        )
    return pyo.Constraint.Skip
m.Q_balance = pyo.Constraint(m.BUS, m.TIME, rule=power_balance_Q)

# |Sij| limit
m.Sij_limit = pyo.ConstraintList()
for k in range(len(linedata)):
    i = int(linedata[k,0]); j = int(linedata[k,1])
    G = Ybus[i-1,j-1].real; B = Ybus[i-1,j-1].imag
    for t in m.TIME:
        Vi, Vj = m.V[i,t], m.V[j,t]
        di, dj = m.delta[i,t], m.delta[j,t]
        Pij = Vi**2 * G - Vi*Vj*(G*pyo.cos(di-dj) + B*pyo.sin(di-dj))
        Qij = -Vi**2 * B - Vi*Vj*(G*pyo.sin(di-dj) - B*pyo.cos(di-dj))
        m.Sij_limit.add(Pij**2 + Qij**2 <= m.Smax[k+1]**2)

# H2 storage constraints
def h2_storage_cap_rule(m, g, t):
    return m.H2_level[g,t] <= m.H2_cap0[g]
m.H2_storage_cap_con = pyo.Constraint(m.H2GEN, m.TIME, rule=h2_storage_cap_rule)

def h2_balance_rule(m, g, t):
    Pg_MW = m.Pg[g,t] * base_mva
    H2_consumption = m.kH2[g] * Pg_MW
    if t == 1:
        return m.H2_level[g,t] == m.H2_init0[g] + m.H2_buy[g,t] - H2_consumption
    else:
        return m.H2_level[g,t] == m.H2_level[g,t-1] + m.H2_buy[g,t] - H2_consumption
m.H2_balance_con = pyo.Constraint(m.H2GEN, m.TIME, rule=h2_balance_rule)

def ramp_rule(m, g, t):
    if t == 1:
        return pyo.inequality(-m.RampDown[g], m.Pg[g,t] - m.Pg_init[g], m.RampUp[g])
    return pyo.inequality(-m.RampDown[g], m.Pg[g,t] - m.Pg[g,t-1], m.RampUp[g])
m.ramp_con = pyo.Constraint(m.GEN, m.TIME, rule=ramp_rule)

print("6. Definisikan Objective Function")
def objective_rule(m):
    gen_cost = sum(m.a[g]*(m.Pg[g,t]*base_mva)**2 + m.b[g]*(m.Pg[g,t]*base_mva) + m.c[g]
                   for g in m.GEN for t in m.TIME)
    h2_buy_cost = sum(m.priceH2[g]*m.H2_buy[g,t] for g in m.H2GEN for t in m.TIME)
    return gen_cost + h2_buy_cost
m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

print("7. Jalankan Optimasi")
solver = pyo.SolverFactory('ipopt')
results = solver.solve(m, tee=True)

print("\n=== Hasil Solusi ===")
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("✅ Solusi optimal ditemukan!")
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("❌ Tidak ada solusi yang memenuhi constraint (infeasible)."); sys.exit(0)
else:
    print(f"⚠️ Solver selesai dengan status: {results.solver.status}, kondisi: {results.solver.termination_condition}"); sys.exit(0)

print("8. Hasil Optimasi")
print("\n=== Output Generator ===")
total_gen_cost = 0.0
total_h2_buy_cost = 0.0

for g in m.GEN:
    for t in m.TIME:
        Pg_MW   = pyo.value(m.Pg[g,t]) * base_mva
        Qg_Mvar = pyo.value(m.Qg[g,t]) * base_mva

        # Thermal generator cost (per period)
        cost_gen = pyo.value(m.a[g])*(Pg_MW**2) + pyo.value(m.b[g])*Pg_MW + pyo.value(m.c[g])
        total_gen_cost += cost_gen

        if g in list(m.H2GEN):
            # Ambil parameter H2 dari model
            kH2     = pyo.value(m.kH2[g])
            cap     = pyo.value(m.H2_cap0[g])
            price   = pyo.value(m.priceH2[g])

            H2_buy   = pyo.value(m.H2_buy[g,t])
            H2_level = pyo.value(m.H2_level[g,t])
            H2_cons  = kH2 * Pg_MW
            h2_cost  = price * H2_buy
            total_h2_buy_cost += h2_cost

            print(f"Gen {g} (H2) | Period {t}")
            print(f"   Pg = {Pg_MW:.3f} MW, Qg = {Qg_Mvar:.3f} Mvar")
            print(f"   H2 consumption = {H2_cons:.2f} kg, "
                  f"H2_buy = {H2_buy:.2f} kg, "
                  f"H2_level = {H2_level:.2f} / {cap:.2f} kg")
            print(f"   Biaya thermal = {cost_gen:.2f} $, Biaya H2 = {h2_cost:.2f} $\n")

        else:
            print(f"Gen {g} (Thermal) | Period {t}")
            print(f"   Pg = {Pg_MW:.3f} MW, Qg = {Qg_Mvar:.3f} Mvar")
            print(f"   Biaya thermal = {cost_gen:.2f} $\n")

# Rekap total biaya
print("=== Rekap Biaya Sistem ===")
print(f"Total Thermal Gen cost : {total_gen_cost:.2f} $")
print(f"Total H2 Purchase cost : {total_h2_buy_cost:.2f} $")
print(f"Total System Cost      : {total_gen_cost + total_h2_buy_cost:.2f} $")

print("\n=== H2 Storage Status ===")
for g in m.H2GEN:
    for t in m.TIME:
        level   = pyo.value(m.H2_level[g,t])
        cap     = pyo.value(m.H2_cap0[g])
        H2_buy  = pyo.value(m.H2_buy[g,t])
        price   = pyo.value(m.priceH2[g])
        Pg_MW   = pyo.value(m.Pg[g,t]) * base_mva   # <-- tambahkan ,t
        consump = pyo.value(m.kH2[g]) * Pg_MW

        print(f"Gen {g} | Period {t}:")
        print(f"   H2 Level   : {level:.2f} / {cap:.2f} kg")
        print(f"   H2 Consump : {consump:.2f} kg")
        print(f"   H2 Buy     : {H2_buy:.2f} kg @ {price:.2f} $/kg")

print("9. Hasil Bus Out")
for t in m.TIME:
    print(f"\n=== Hasil Bus Out (Period {t}) ===")
    Vm = np.array([value(m.V[i,t]) for i in m.BUS])
    delta_rad = np.array([value(m.delta[i,t]) for i in m.BUS])
    delta_deg = np.degrees(delta_rad)

    Pg_arr = np.zeros(nbus); Qg_arr = np.zeros(nbus)
    for g in m.GEN:
        b = m.gen_bus[g]
        Pg_arr[b-1] += value(m.Pg[g,t]) * base_mva
        Qg_arr[b-1] += value(m.Qg[g,t]) * base_mva

    busdata_np = np.array([
        [i, 1 if bus_data[i]['type']=='Slack' else 2 if bus_data[i]['type']=='PV' else 0,
         bus_data[i]['V'], 0, bus_data[i]['Pd'], bus_data[i]['Qd'], 0, 0,
         bus_data[i]['Vmin'], bus_data[i]['Vmax'], 0]
        for i in bus_data
    ])
    busout(busdata_np, Vm, delta_deg, Pg_arr, Qg_arr)

    print(f"\n=== Hasil Line Flow (Period {t}) ===")
    V_complex = Vm * np.exp(1j * delta_rad)
    line_results, SLT = lineflow(linedata, V_complex, base_mva)

    for res in line_results:
        f,tbus = res['from'], res['to']
        Snk,Skn,SL = res['Snk'], res['Skn'], res['SL']
        print(f"Line {f} → {tbus}:")
        print(f"  S{f}{tbus} = {Snk.real:.4f} + j{Snk.imag:.4f} MVA")
        print(f"  S{tbus}{f} = {Skn.real:.4f} + j{Skn.imag:.4f} MVA")
        print(f"  Loss    = {SL.real:.4f} + j{SL.imag:.4f} MVA\n")

    total_P_loss = sum((res['Snk'] + res['Skn']).real for res in line_results)
    total_Q_loss = sum((res['Snk'] + res['Skn']).imag for res in line_results)
    print(f"Total system losses (Period {t}): {total_P_loss:.4f} + j{total_Q_loss:.4f} MVA")

    print("\n🔎 Cek Magnitude Line Flow:")
    for res in line_results:
        Sij_mag = abs(res['Snk'])
        print(f"Line {res['from']} → {res['to']}: |Sij| = {Sij_mag:.4f} MVA")