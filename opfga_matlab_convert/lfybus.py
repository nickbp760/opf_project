# lfybus.py
import numpy as np

def build_ybus(linedata, nbus):
    """
    Membentuk matriks Ybus dari linedata.
    """
    j = 1j
    nl = linedata[:, 0].astype(int) # from bus
    nr = linedata[:, 1].astype(int) # to bus
    R = linedata[:, 2]
    X = linedata[:, 3]
    Bc = j * linedata[:, 4]
    a = linedata[:, 5]

    nbr = len(nl)
    # Ini menghitung impedansi saluran dan admitansi saluran.
    Z = R + 1j * X
    y = 1 / Z
    # Membuat matriks Ybus kosong, ukuran [nbus x nbus].
    Ybus = np.zeros((nbus, nbus), dtype=complex)

    # Jika tap ratio <= 0, ganti jadi 1
    # Karena jika a=0 artinya tidak ada trafo.
    a[a <= 0] = 1

    # Isi Off-diagonal Elemen
    # Setiap koneksi saluran antara bus i dan j
    # diisi nilai negatif dari admitansi saluran. Ini berlaku simetris.
    # Kenapa minus?
    # Karena dari sudut pandang node i, aliran ke node j = keluar → nilainya negatif.
    for k in range(nbr):
        i = nl[k] - 1
        j_ = nr[k] - 1
        Ybus[i, j_] -= y[k] / a[k]
        Ybus[j_, i] = Ybus[i, j_]  # Symmetric

    # Isi Diagonal Elemen
    # Mengisi nilai total admitansi masuk ke tiap bus:
    # dari saluran keluar (diperhatikan tap)
    # dan dari saluran masuk
    # serta menambahkan kapasitor (Bc) ke diagonal
    for n in range(nbus):
        for k in range(nbr):
            if nl[k] == n + 1:
                Ybus[n, n] += y[k] / (a[k]**2) + Bc[k]
            elif nr[k] == n + 1:
                Ybus[n, n] += y[k] + Bc[k]

    return Ybus


# linedata = np.array([
#   [1, 2, 0.02, 0.06, 0.03, 1],
#   [1, 3, 0.08, 0.24, 0.025, 1],
#   [2, 3, 0.06, 0.18, 0.02, 1]
# ])
# nbus = 3
# Ybus = build_ybus(linedata, nbus)
# print(Ybus)

# Penjelasan Gampang Setiap Parameter
# 1. R — Resistansi (Ω)
# Ibarat kabel itu kawat panjang.
# Semakin tinggi R, semakin banyak energi hilang jadi panas.
# Ini menyebabkan rugi-rugi daya.

# 2. X — Reaktansi (Ω)
# Karena listrik AC (bolak-balik), arus ditahan oleh medan magnet.
# X merepresentasikan itu — semakin tinggi X, semakin sulit arus lewat.

# 3. Bc — Shunt Capacitance (siemens)
# Beberapa kabel panjang seperti menyimpan muatan listrik.
# Nilai ini mewakili efek “kapasitif” kabel.
# Biasanya kecil, tapi berpengaruh di sistem besar.

# 4. a — Tap Ratio (transformer)
# Kalau jalur ini lewat trafo, bisa ada perubahan tegangan.
# Misalnya a = 1 artinya tidak ada trafo.
# a ≠ 1 → berarti saluran ini berada di sisi trafo, dan arus/tegangan perlu disesuaikan.

# Apa itu Matriks Ybus Lagi?
# Ybus = Matriks admitansi berukuran nbus × nbus
# Setiap elemen 𝑌 𝑖 𝑗 Y ij ​ adalah admitansi dari bus i ke bus j
# Kalau 𝑖 = 𝑗 elemen diagonal → hubungan dalam bus itu sendiri
# Kalau 𝑖 ≠ 𝑗 i elemen off-diagonal → hubungan antar bus
