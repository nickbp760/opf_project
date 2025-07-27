import numpy as np

def busout(busdata, Vm, delta_deg, Pg, Qg):
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

if __name__ == "__main__":
    base_mva = 100.0

    # Busdata IEEE 3-bus
    busdata = np.array([
        [1, 1, 1.06, 0,    0,   0,   0,   0, -999, 999, 0],   # Slack
        [2, 2, 1.00, 0,  100,  60, 150,   0, -999, 999, 0],   # PV
        [3, 0, 1.00, 0,   90,  30,   0,   0, -999, 999, 0],   # PQ
    ])

    # Load flow results
    Vm = np.array([1.0600, 1.0000, 0.9994])
    delta_deg = np.array([0.0000, 1.8130, -0.7901])
    Pg = np.array([45.298, 150.0, 0.0])    # in MW
    Qg = np.array([137.9138, -37.3178, 0.0001])  # in MVar

    busout(busdata, Vm, delta_deg, Pg, Qg)
