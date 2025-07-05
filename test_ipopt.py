from pyomo.environ import SolverFactory
print(SolverFactory('ipopt').available())  # Harusnya True

import sys
print(sys.executable)
