import numpy as np
from lfybus import build_ybus
from lfnewton import lfnewton
from lineflow import lineflow
from busout import busout

# ----------------------------
# 1. DATA INPUT
# ----------------------------
linedata = np.array([
    [1, 2, 0.02, 0.06, 0.03, 1],
    [1, 3, 0.08, 0.24, 0.025, 1],
    [2, 3, 0.06, 0.18, 0.02, 1],
])

busdata = np.array([
    [1, 1, 1.06, 0,     0,    0,   232.4, -16.9, 0, 0, 0],
    [2, 2, 1.045, 0,   21.7, 12.7, 40,    42.4,  0, 0, 0],
    [3, 0, 1.01,  0,   94.2, 19.0, 0,     0,     0, 0, 0],
])

base_mva = 100

# ----------------------------
# 2. HITUNG Ybus
# ----------------------------
nbus = busdata.shape[0]
Ybus = build_ybus(linedata, nbus)

# ----------------------------
# 3. JALANKAN POWER FLOW
# ----------------------------
Vm, delta_deg, Pg, Qg, S, P_calc, Q_calc = lfnewton(busdata, Ybus, base_mva)

# ----------------------------
# 4. CETAK HASIL
# ----------------------------
busout(busdata, Vm, delta_deg, Pg, Qg, P_calc, Q_calc)

# ----------------------------
# 5. HITUNG LINE FLOW
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
