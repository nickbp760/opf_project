import numpy as np

def busout(busdata, Vm, delta_deg, Pg, Qg, P_calc, Q_calc):
    nbus = busdata.shape[0]
    Pd = busdata[:, 4]
    Qd = busdata[:, 5]

    Pdt = np.sum(Pd)
    Qdt = np.sum(Qd)
    Pgt = np.sum(Pg)
    Qgt = np.sum(Qg)

    print("\nSETELAH LOADFLOW\n")
    print(f"{'Bus':>5} {'Voltage':>8} {'Angle':>8}    ------Load------    ---Generation---")
    print(f"{'No.':>5} {'Mag.':>8} {'Degree':>8}     kW       kvar       kW       kvar")

    for n in range(nbus):
        print(f"{n+1:5d} {Vm[n]:8.3f} {delta_deg[n]:8.3f} {Pd[n]:10.3f} {Qd[n]:10.3f} {Pg[n]:10.3f} {Qg[n]:10.3f}")

    print("\n    Total")
    print(f"{'':25s}{Pdt:10.3f} {Qdt:10.3f} {Pgt:10.3f} {Qgt:10.3f}\n")
