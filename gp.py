from gplearn.genetic import SymbolicRegressor
import numpy as np
import parser
import sys

x_train, y_train, x_test, y_test = parser.uci_data(sys.argv[1])

gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

gp.fit(x_train, y_train)

print(gp._program)


'''

    RESULTS SO FAR FOR FEATURE SELECTION:

    WINE: X2, X6, X11


'''
