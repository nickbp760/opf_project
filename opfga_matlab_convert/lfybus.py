# lfybus.py
import numpy as np

def build_ybus(linedata, nbus):
    """
    Membentuk matriks Ybus dari linedata.
    """
    j = 1j
    nl = linedata[:, 0].astype(int)
    nr = linedata[:, 1].astype(int)
    R = linedata[:, 2]
    X = linedata[:, 3]
    Bc = j * linedata[:, 4]
    a = linedata[:, 5]

    nbr = len(nl)
    Z = R + 1j * X
    y = 1 / Z
    Ybus = np.zeros((nbus, nbus), dtype=complex)

    # Jika tap ratio <= 0, ganti jadi 1
    a[a <= 0] = 1

    for k in range(nbr):
        i = nl[k] - 1
        j_ = nr[k] - 1
        Ybus[i, j_] -= y[k] / a[k]
        Ybus[j_, i] = Ybus[i, j_]  # Symmetric

    for n in range(nbus):
        for k in range(nbr):
            if nl[k] == n + 1:
                Ybus[n, n] += y[k] / (a[k]**2) + Bc[k]
            elif nr[k] == n + 1:
                Ybus[n, n] += y[k] + Bc[k]

    return Ybus
