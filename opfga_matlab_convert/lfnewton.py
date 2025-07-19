import numpy as np

def lfnewton(busdata, Ybus, base_mva=100, accuracy=1e-6, max_iter=10):
    j = 1j
    nbus = busdata.shape[0]

    # Inisialisasi
    Vm = busdata[:, 2].copy()
    delta = np.radians(busdata[:, 3].copy())
    Pd = busdata[:, 4]
    Qd = busdata[:, 5]
    Pg = busdata[:, 6]
    Qg = busdata[:, 7]
    Qmin = busdata[:, 8]
    Qmax = busdata[:, 9]
    Qsh = busdata[:, 10]

    kb = busdata[:, 1].astype(int)
    V = Vm * np.exp(j * delta)
    P = (Pg - Pd) / base_mva
    Q = (Qg - Qd + Qsh) / base_mva

    swing_idx = np.where(kb == 1)[0]
    pv_idx = np.where(kb == 2)[0]
    pq_idx = np.where(kb == 0)[0]

    converge = True
    for iteration in range(max_iter):
        V = Vm * np.exp(j * delta)
        I = Ybus @ V
        S_calc = V * np.conj(I)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        dP = P - P_calc
        dQ = Q - Q_calc

        mismatch = []
        for i in range(nbus):
            if kb[i] != 1:
                mismatch.append(dP[i])
        for i in pq_idx:
            mismatch.append(dQ[i])
        mismatch = np.array(mismatch)

        maxerror = np.max(np.abs(mismatch))
        if maxerror < accuracy:
            break

        # Hitung Jacobian
        J11 = np.zeros((nbus, nbus))
        J12 = np.zeros((nbus, nbus))
        J21 = np.zeros((nbus, nbus))
        J22 = np.zeros((nbus, nbus))

        for i in range(nbus):
            for k in range(nbus):
                if i == k:
                    for m in range(nbus):
                        VmVmY = Vm[i] * Vm[m] * abs(Ybus[i, m])
                        theta = np.angle(Ybus[i, m])
                        J11[i, i] += VmVmY * np.sin(theta + delta[m] - delta[i])
                        if kb[i] == 0:
                            J22[i, i] += Vm[m] * abs(Ybus[i, m]) * np.cos(theta + delta[m] - delta[i])
                    J11[i, i] *= -1
                    J22[i, i] *= 2 * Vm[i]
                else:
                    VmVmY = Vm[i] * Vm[k] * abs(Ybus[i, k])
                    theta = np.angle(Ybus[i, k])
                    J11[i, k] = VmVmY * np.sin(theta + delta[k] - delta[i])
                    J12[i, k] = Vm[i] * abs(Ybus[i, k]) * np.cos(theta + delta[k] - delta[i])
                    J21[i, k] = -VmVmY * np.cos(theta + delta[k] - delta[i])
                    J22[i, k] = Vm[i] * abs(Ybus[i, k]) * np.sin(theta + delta[k] - delta[i])

        # Ambil bagian yang sesuai untuk variabel bukan swing bus
        pv_pq = np.concatenate((pv_idx, pq_idx))
        J1 = J11[np.ix_(pv_pq, pv_pq)]
        J2 = J12[np.ix_(pv_pq, pq_idx)]
        J3 = J21[np.ix_(pq_idx, pv_pq)]
        J4 = J22[np.ix_(pq_idx, pq_idx)]

        J = np.block([[J1, J2], [J3, J4]])

        dx = np.linalg.solve(J, mismatch)

        # Update variabel
        for idx, i in enumerate(pv_pq):
            delta[i] += dx[idx]
        for idx, i in enumerate(pq_idx):
            Vm[i] += dx[len(pv_pq) + idx]

    else:
        converge = False

    if not converge:
        print("WARNING: Load flow tidak konvergen")

    V = Vm * np.exp(j * delta)
    I = Ybus @ V
    S_calc = V * np.conj(I)
    Pg_new = S_calc.real * base_mva + Pd
    Qg_new = S_calc.imag * base_mva + Qd - Qsh

    return Vm, np.degrees(delta), Pg_new, Qg_new, S_calc, S_calc.real, S_calc.imag
