import numpy as np

def lineflow(linedata, V, base_mva=100):
    j = 1j
    nl = linedata[:, 0].astype(int) - 1  # from bus (0-based index)
    nr = linedata[:, 1].astype(int) - 1  # to bus
    R = linedata[:, 2]
    X = linedata[:, 3]
    Bc = linedata[:, 4]
    a = linedata[:, 5]

    nbr = len(nl)
    Z = R + 1j * X
    y = 1 / Z
    b = j * Bc
    a[a <= 0] = 1

    SLT = 0
    results = []

    for L in range(nbr):
        n = nl[L]
        k = nr[L]

        tap = a[L]
        Y = y[L]
        B = b[L]

        In = (V[n] - V[k] / tap) * Y + B * V[n] / 2
        Ik = (V[k] - V[n] * tap) * Y + B * V[k] / 2

        Snk = V[n] * np.conj(In) * base_mva
        Skn = V[k] * np.conj(Ik) * base_mva

        # Snk = 1.25 + j0.55
        # Skn = -1.22 - j0.53
        # SL  = 0.03 + j0.02   ← losses
        SL = Snk + Skn
        SLT += SL

        results.append({
            'from': n + 1,
            'to': k + 1,
            # Daya kompleks dari bus n ke k (n→k)
            'Snk': Snk,
            # Daya kompleks dari bus k ke n (k→n)
            'Skn': Skn,
            'SL': SL,
            'tap': tap,
        })

    # Total system loss
    SLT = SLT / 2

    return results, SLT


if __name__ == "__main__":
    # V hasil dari lfnewton
    Vm = [1.0600, 1.0000, 0.9994]
    delta_deg = [0.0000, 1.8130, -0.7901]
    delta = np.radians(delta_deg)
    V = Vm * np.exp(1j * delta)

    # Line data
    linedata = np.array([
        [1, 2, 0.02, 0.04, 0.00, 1.0],
        [1, 3, 0.01, 0.03, 0.00, 1.0],
        [2, 3, 0.0125, 0.025, 0.00, 1.0],
    ])

    results, SLT = lineflow(linedata, V)

    # Print
    print("\nHASIL LINE FLOW:")
    print("----------------")
    for res in results:
        f, t = res['from'], res['to']
        Snk, Skn, SL = res['Snk'], res['Skn'], res['SL']
        print(f"Line {f} → {t}:")
        print(f"  S{f}{t} = {Snk.real:.4f} + j{Snk.imag:.4f} MVA")
        print(f"  S{t}{f} = {Skn.real:.4f} + j{Skn.imag:.4f} MVA")
        print(f"  Loss    = {SL.real:.4f} + j{SL.imag:.4f} MVA\n")

        print(f"Total system losses: {SLT.real:.4f} + j{SLT.imag:.4f} MVA")
