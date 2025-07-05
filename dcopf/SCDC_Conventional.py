import pyomo.environ as pyo
import numpy as np
import pandas as pd


def create_dcopf_model():
    """
    Builds the complete Pyomo model for the 118-bus DC Optimal Power Flow problem,
    """
    # Create the concrete model
    model = pyo.ConcreteModel(name="DCOPF_118_Bus_System")

    # =============================================================================
    # DATA LOADING FROM EXCEL
    # =============================================================================
    print("0. Loading data from dcopf_data.xlsx...")
    data_file = 'dcopf_data.xlsx'

    # Load data from each sheet into pandas DataFrames
    branch_df = pd.read_excel(data_file, sheet_name='branch_data')
    gb_df = pd.read_excel(data_file, sheet_name='generator_bus_map')
    gen_df = pd.read_excel(data_file, sheet_name='generator_data')
    re_df = pd.read_excel(data_file, sheet_name='renewable_data')
    bus_df = pd.read_excel(data_file, sheet_name='bus_data')

    # Convert DataFrames to dictionaries for Pyomo initialization
    branch_data_raw = branch_df.set_index(['from_bus', 'to_bus']).to_dict('index')
    gb_data = {tuple(x): 1 for x in gb_df.to_numpy()}
    gen_data = gen_df.set_index('generator_id').to_dict('index')
    re_data_raw = re_df.set_index('time_period').to_dict('index')

    # For bus data, we create separate dictionaries for each parameter
    bus_demand_raw = bus_df.set_index('bus')['pd'].to_dict()
    # =============================================================================
    # SETS
    # =============================================================================
    print("1. Defining Sets...")
    model.bus = pyo.Set(initialize=bus_df['bus'].tolist())
    model.Gen = pyo.Set(initialize=gen_df['generator_id'].tolist())
    model.t = pyo.Set(initialize=re_df['time_period'].tolist(), ordered=True)
    model.k = pyo.Set(initialize=[f'sg{i}' for i in range(1, 101)], ordered=True)
    model.slack = pyo.Set(initialize=[69])

    # Derived Sets
    model.branch = pyo.Set(initialize=branch_data_raw.keys(), dimen=2)
    model.GB = pyo.Set(initialize=gb_data.keys(), dimen=2)

    # =============================================================================
    # PARAMETERS
    # =============================================================================
    print("2. Defining Parameters...")
    model.Sbase = pyo.Param(initialize=100)

    # Generator Data
    model.GD = pyo.Param(model.Gen, ['c', 'b', 'a', 'Pmax', 'Pmin', 'RU', 'RD'],
                         initialize=lambda m, g, p: gen_data[g][p])

    # Bus Demand Data
    model.BusData_pd = pyo.Param(model.bus, initialize=bus_demand_raw, default=0)

    # Branch Data
    model.branch_x = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['x'])
    model.branch_Limit = pyo.Param(model.branch, initialize=lambda m, i, j: branch_data_raw[(i, j)]['Limit'])
    model.bij = pyo.Param(model.branch,
                          initialize=lambda m, i, j: 1 / model.branch_x[i, j] if model.branch_x[i, j] != 0 else 0)

    # Renewable Energy Data
    model.RE = pyo.Param(model.t, ['d'],
                         initialize=lambda m, t, p: re_data_raw[t][p])

    # Calculated Parameters for Piecewise Linear Cost Function
    print("3. Calculating Piecewise Linear Cost Parameters...")
    piecewise_data = {}
    mincost_data = {}
    for g in model.Gen:
        p_min = model.GD[g, 'Pmin']
        p_max = model.GD[g, 'Pmax']
        a = model.GD[g, 'a']
        b = model.GD[g, 'b']
        c = model.GD[g, 'c']

        mincost_data[g] = a * p_min ** 2 + b * p_min + c

        dp = (p_max - p_min) / len(model.k)
        for seg_idx, seg in enumerate(model.k):
            p_ini = p_min + seg_idx * dp
            p_fin = p_min + (seg_idx + 1) * dp
            c_ini = a * p_ini ** 2 + b * p_ini + c
            c_fin = a * p_fin ** 2 + b * p_fin + c
            slope = (c_fin - c_ini) / dp if dp > 0 else 0
            piecewise_data[g, seg] = {'s': slope, 'DP': dp}

    model.Mincost = pyo.Param(model.Gen, initialize=mincost_data)
    model.Slope = pyo.Param(model.Gen, model.k, initialize=lambda m, g, seg: piecewise_data[g, seg]['s'])
    model.SegWidth = pyo.Param(model.Gen, model.k, initialize=lambda m, g, seg: piecewise_data[g, seg]['DP'])

    # =============================================================================
    # VARIABLES
    # =============================================================================
    print("4. Defining Variables...")
    model.OF = pyo.Var(domain=pyo.Reals)
    model.Pij = pyo.Var(model.branch, model.t, domain=pyo.Reals)
    model.Pg = pyo.Var(model.Gen, model.t, domain=pyo.NonNegativeReals)
    model.delta = pyo.Var(model.bus, model.t, domain=pyo.Reals, bounds=(-np.pi, np.pi))
    model.costThermal = pyo.Var(model.t, domain=pyo.Reals)
    model.Pk = pyo.Var(model.Gen, model.t, model.k, domain=pyo.NonNegativeReals)

    # =============================================================================
    # CONSTRAINTS
    # =============================================================================
    print("5. Defining Constraints...")

    # const1: DC Power Flow equation
    @model.Constraint(model.branch, model.t)
    def const1(m, i, j, t):
        return m.Pij[i, j, t] == m.bij[i, j] * (m.delta[i, t] - m.delta[j, t])

    # const2: Power Balance at each bus
    @model.Constraint(model.bus, model.t)
    def const2(m, b, t):
        gen_at_bus = sum(m.Pg[g, t] for g in m.Gen if (b, g) in m.GB)
        load_at_bus = (m.RE[t, 'd'] / 4242) * m.BusData_pd[b] / m.Sbase
        flow_out = sum(m.Pij[b, j, t] for j in m.bus if (b, j) in m.branch)
        flow_in = sum(m.Pij[j, b, t] for j in m.bus if (j, b) in m.branch)

        return (
                    gen_at_bus - load_at_bus) == (
                    flow_out - flow_in)

    # const4 & const5: Ramping limits
    @model.Constraint(model.Gen, model.t)
    def const4(m, g, t):  # Ramp-up
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.Pg[g, t] - m.Pg[g, m.t.prev(t)] <= m.GD[g, 'RU'] / m.Sbase

    @model.Constraint(model.Gen, model.t)
    def const5(m, g, t):  # Ramp-down
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.Pg[g, m.t.prev(t)] - m.Pg[g, t] <= m.GD[g, 'RD'] / m.Sbase

    # const7: Thermal cost calculation per hour
    @model.Constraint(model.t)
    def const7(m, t):
        return m.costThermal[t] == sum(
            m.Mincost[g] + sum(m.Slope[g, seg] * m.Pk[g, t, seg] for seg in m.k) for g in m.Gen)

    # const8: Generator output defined by piecewise segments
    @model.Constraint(model.Gen, model.t)
    def const8(m, g, t):
        return m.Pg[g, t] * m.Sbase == m.GD[g, 'Pmin'] + sum(m.Pk[g, t, seg] for seg in m.k)

    # =============================================================================
    # VARIABLE BOUNDS
    # =============================================================================
    print("6. Setting Variable Bounds...")

    # Generator Limits
    @model.Constraint(model.Gen, model.t)
    def pg_bounds_up(m, g, t):
        return m.Pg[g, t] <= m.GD[g, 'Pmax'] / m.Sbase

    @model.Constraint(model.Gen, model.t)
    def pg_bounds_lo(m, g, t):
        return m.Pg[g, t] >= m.GD[g, 'Pmin'] / m.Sbase

    # Slack bus angle fixed to 0
    for t in model.t:
        model.delta[69, t].fix(0)

    # Power flow limits on branches
    @model.Constraint(model.branch, model.t)
    def pij_bounds(m, i, j, t):
        limit = m.branch_Limit[i, j] / m.Sbase
        return pyo.inequality(-limit, m.Pij[i, j, t], limit)

   # Piecewise segment variable bounds
    @model.Constraint(model.Gen, model.t, model.k)
    def pk_bounds(m, g, t, seg):
        return m.Pk[g, t, seg] <= m.SegWidth[g, seg]

    # =============================================================================
    # OBJECTIVE FUNCTION
    # =============================================================================
    print("7. Defining Objective Function...")

    # const3: This constraint now correctly defines the OF variable.
    @model.Constraint()
    def const3(m):
        return m.OF == sum(m.costThermal[t] for t in m.t)

    # This line now correctly defines the objective function for the model.
    model.objective = pyo.Objective(expr=model.OF, sense=pyo.minimize)

    return model


if __name__ == '__main__':
    # 1. Create the Pyomo model instance
    dcopf_model = create_dcopf_model()

    # 2. Create the Gurobi solver instance
    # This assumes that the Gurobi executable is in your system's PATH.
    # If not, you can specify the path using the 'executable' argument:
    # solver = pyo.SolverFactory('gurobi', executable='/path/to/gurobi.exe')
    print("\n8. Creating Gurobi solver instance...")
    solver = pyo.SolverFactory('ipopt')

    # You can set specific Gurobi options if needed, for example:
    # solver.options['MIPGap'] = 0.01
    # solver.options['TimeLimit'] = 120 # Time limit in seconds

    # 3. Solve the model
    # The 'tee=True' argument will stream the solver's log to the console.
    print("9. Solving the model with Gurobi...")
    results = solver.solve(dcopf_model, tee=True)

    # 4. Display the results
    print("\n" + "=" * 50)
    print("SOLVER RESULTS")
    print("=" * 50)
    print(f"Solver Status: {results.solver.status}")
    print(f"Termination Condition: {results.solver.termination_condition}")

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print(f"Optimal Total Cost: ${pyo.value(dcopf_model.objective):,.2f}")
    else:
        print("Could not find an optimal solution.")

    # You can also inspect variable values after solving, for example:
    # print("\nFirst hour generation for generator g1:")
    # print(f"Pg['g1', 't1'] = {pyo.value(dcopf_model.Pg['g1', 't1'])}")