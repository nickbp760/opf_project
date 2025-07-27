import numpy as np

def lfnewton(busdata, Ybus, base_mva=100.0, tol=1e-6, max_iter=100):
    j = 1j
    nbus = busdata.shape[0]
    bus_type = busdata[:, 1].astype(int)
    
    # Initial values
    Vm = busdata[:, 2].copy()
    delta = np.radians(busdata[:, 3].copy())

    Pd = busdata[:, 4] / base_mva
    Qd = busdata[:, 5] / base_mva
    Pg = busdata[:, 6] / base_mva
    Qg = busdata[:, 7] / base_mva
    Qsh = busdata[:, 10] / base_mva

    P_spec = Pg - Pd
    Q_spec = Qg - Qd + Qsh

    pq = np.where(bus_type == 0)[0]
    pv = np.where(bus_type == 2)[0]
    slack = np.where(bus_type == 1)[0][0]

    npq = len(pq)
    npv = len(pv)

    for iteration in range(max_iter):
        V = Vm * np.exp(1j * delta)
        S = V * np.conj(Ybus @ V)
        P = S.real
        Q = S.imag

        dP = P_spec - P
        dQ = Q_spec - Q

        mismatch = np.concatenate((dP[np.r_[pv, pq]], dQ[pq]))
        if np.max(np.abs(mismatch)) < tol:
            break

        # Jacobian
        J11 = np.zeros((npv + npq, npv + npq))
        J12 = np.zeros((npv + npq, npq))
        J21 = np.zeros((npq, npv + npq))
        J22 = np.zeros((npq, npq))

        for i in range(nbus):
            for k in range(nbus):
                Gik = Ybus[i, k].real
                Bik = Ybus[i, k].imag
                if i == k:
                    Vi = Vm[i]
                    for m in range(nbus):
                        if m == i:
                            continue
                        Vm_ = Vm[m]
                        angle = delta[i] - delta[m]
                        G = Ybus[i, m].real
                        B = Ybus[i, m].imag

                        if i in np.r_[pv, pq]:
                            idx = np.where(np.r_[pv, pq] == i)[0][0]
                            J11[idx, idx] += Vm[i]*Vm_*(-G*np.sin(angle)+B*np.cos(angle))
                        if i in pq:
                            idx = np.where(pq == i)[0][0]
                            J21[idx, idx] += Vm[i]*Vm_*(G*np.cos(angle)+B*np.sin(angle))

                    if i in np.r_[pv, pq]:
                        idx_row = np.where(np.r_[pv, pq] == i)[0][0]
                        for m in pq:
                            angle = delta[i] - delta[m]
                            G = Ybus[i, m].real
                            B = Ybus[i, m].imag
                            idx_col = np.where(pq == m)[0][0]
                            J12[idx_row, idx_col] = Vm[i]*(G*np.cos(angle)+B*np.sin(angle)) + (i == m)*2*Vm[i]*Ybus[i, i].real

                    if i in pq:
                        idx_row = np.where(pq == i)[0][0]
                        for m in pq:
                            angle = delta[i] - delta[m]
                            G = Ybus[i, m].real
                            B = Ybus[i, m].imag
                            idx_col = np.where(pq == m)[0][0]
                            J22[idx_row, idx_col] = Vm[i]*(G*np.sin(angle)-B*np.cos(angle))
                            if i == m:
                                J22[idx_row, idx_col] -= 2*Vm[i]*Ybus[i, i].imag

                else:
                    angle = delta[i] - delta[k]
                    Vi = Vm[i]
                    Vk = Vm[k]
                    G = Gik
                    B = Bik

                    if i in np.r_[pv, pq] and k in np.r_[pv, pq]:
                        idx_i = np.where(np.r_[pv, pq] == i)[0][0]
                        idx_k = np.where(np.r_[pv, pq] == k)[0][0]
                        J11[idx_i, idx_k] = Vi*Vk*(G*np.sin(angle)-B*np.cos(angle))

                    if i in pq and k in np.r_[pv, pq]:
                        idx_i = np.where(pq == i)[0][0]
                        idx_k = np.where(np.r_[pv, pq] == k)[0][0]
                        J21[idx_i, idx_k] = -Vi*Vk*(G*np.cos(angle)+B*np.sin(angle))

                    if i in np.r_[pv, pq] and k in pq:
                        idx_i = np.where(np.r_[pv, pq] == i)[0][0]
                        idx_k = np.where(pq == k)[0][0]
                        J12[idx_i, idx_k] = Vi*(G*np.cos(angle)+B*np.sin(angle))

                    if i in pq and k in pq:
                        idx_i = np.where(pq == i)[0][0]
                        idx_k = np.where(pq == k)[0][0]
                        J22[idx_i, idx_k] = Vi*(G*np.sin(angle)-B*np.cos(angle))

        J = np.block([[J11, J12], [J21, J22]])
        dx = np.linalg.solve(J, mismatch)

        d_delta = dx[:npv + npq]
        d_Vm = dx[npv + npq:]

        for i, idx in enumerate(np.r_[pv, pq]):
            delta[idx] += d_delta[i]
        for i, idx in enumerate(pq):
            Vm[idx] += d_Vm[i]
    else:
        print("⚠️ Tidak konvergen dalam", max_iter, "iterasi")

    V = Vm * np.exp(1j * delta)
    Sbus = V * np.conj(Ybus @ V)
    Pg_new = Sbus.real * base_mva + Pd * base_mva
    Qg_new = Sbus.imag * base_mva + Qd * base_mva - Qsh * base_mva

    return Vm, np.degrees(delta), Pg_new, Qg_new, Sbus


if __name__ == "__main__":
    # Data bus IEEE 3 Bus: [No, Type, Vm, delta, Pd, Qd, Pg, Qg, Qmin, Qmax, Qsh]
    # Type: 0 = PQ, 1 = Slack, 2 = PV
    busdata = np.array([
        [1, 1, 1.06, 0,    0,   0,   0,   0, -999, 999, 0],   # Slack bus
        [2, 2, 1.00, 0,  100,  60, 150,   0, -999, 999, 0],   # PV bus
        [3, 0, 1.00, 0,   90,  30,   0,   0, -999, 999, 0],   # PQ bus
    ])

    # Ybus untuk sistem IEEE 3-bus
    Ybus = np.array([
        [10 - 20j, -5 + 10j, -5 + 10j],
        [-5 + 10j, 10 - 20j, -5 + 10j],
        [-5 + 10j, -5 + 10j, 10 - 20j],
    ])

    # Jalankan load flow
    Vm, delta_deg, Pg_new, Qg_new, Sbus = lfnewton(busdata, Ybus)

    # Cetak hasil
    print("\nHASIL LOAD FLOW (IEEE 3 BUS)")
    print("-----------------------------")
    for i in range(len(Vm)):
        print(f"Bus {i+1}: Vm = {Vm[i]:.4f} pu, δ = {delta_deg[i]:.4f}°")
    print("\nPembangkit Aktif (Pg):", np.round(Pg_new, 4))
    print("Pembangkit Reaktif (Qg):", np.round(Qg_new, 4))
    print("\nDaya Kompleks pada Bus (Sbus):", np.round(Sbus, 4))
