import numpy as np

def build_ybus(linedata, nbus):
    j = 1j
    Ybus = np.zeros((nbus, nbus), dtype=complex)

    for row in linedata:
        from_bus = int(row[0]) - 1
        to_bus = int(row[1]) - 1
        R = row[2]
        X = row[3]
        B_half = row[4]  # Ini memang sudah B/2 dari data
        tap = row[5] if row[5] != 0 else 1

        Z = complex(R, X)
        y = 1 / Z
        y_shunt = j * B_half  # ✅ Tidak perlu dibagi 2 lagi

        # Diagonal
        Ybus[from_bus, from_bus] += (y + y_shunt) / (tap ** 2)
        Ybus[to_bus, to_bus]     += y + y_shunt
        # Off-diagonal
        Ybus[from_bus, to_bus]  -= y / tap
        Ybus[to_bus, from_bus]  -= y / tap

    return Ybus


if __name__ == "__main__":
    # Contoh 1 line saja antara Bus 1 dan 2
    # Format: [From, To, R, X, B/2, Tap]
    linedata = np.array([
        [1, 2, 0.02, 0.06, 0.015, 1]  # total B = 0.03
    ])
    nbus = 2
    Ybus = build_ybus(linedata, nbus)
    np.set_printoptions(precision=4, suppress=True)
    print("Ybus Matrix (1-line, final):")
    print(Ybus)
    # Ybus Matrix (1-line, final):
    # [[ 5.-14.985j -5.+15.j   ]
    # [-5.+15.j     5.-14.985j]]

    linedata = np.array([
        [1, 2, 0.02, 0.06, 0.03, 1],     # Line 1–2
        [1, 3, 0.08, 0.24, 0.025, 1],    # Line 1–3
        [2, 3, 0.06, 0.18, 0.02, 1]      # Line 2–3
    ])
    nbus = 3
    Ybus = build_ybus(linedata, nbus)
    np.set_printoptions(precision=4, suppress=True)
    print("Ybus Matrix (3-bus):")
    print(Ybus)
    # Ybus Matrix (3-bus):
    # [[ 6.25  -18.695j -5.    +15.j    -1.25   +3.75j ]
    # [-5.    +15.j     6.6667-19.95j  -1.6667 +5.j   ]
    # [-1.25   +3.75j  -1.6667 +5.j     2.9167 -8.705j]]

