import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

# Read in data for a classifier
# - load data, make some feature eng,  split into train test
Xarr = load_iris()['data']
# Xarr: immer 4 Werte gehören zusammen
yarr = load_iris()['target']

Xint_ = np.array([np.argwhere(np.unique(Xarr) == x).ravel()[0] for x in Xarr.ravel()]) # Xarr: immer 4 Werte gehören zusammen, durch das Xarr.ravel() gehören immer 4 subsequente Werte zu einem Featurevektor
print (Xint_)
Xint = Xint_.reshape(int(Xint_.shape[0]/4),4)
print (Xint)

yint = [np.argwhere(np.unique(yarr) == y).ravel()[0] for y in yarr]
# definition of xcs scenario







import xcs
from xcs.scenarios import Scenario, ScenarioObserver
from xcs.bitstrings import BitString
from xcs import XCSAlgorithm


class ClassifyProblem(Scenario):
    def __init__(self, X, y):
        self.possible_actions = np.unique(y)
        self.initial_training_cycles = len(X)
        self.remaining_cycles = self.initial_training_cycles
        self.target_value = None
        self.X = X
        self.y = y
        self.bits = int(np.log2(np.max(X))) + 1

    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.remaining_cycles = self.initial_training_cycles

    def more(self):
        return self.remaining_cycles > 0

    def sense(self):
        base = [BitString(a, self.bits) for a in self.X[self.remaining_cycles-1].tolist()]
        haystack = BitString(0)
        for b in base:
            haystack += BitString(b)
        self.target_value = self.y[self.remaining_cycles-1] #not bitstringed
        return haystack

    def execute(self, action):
        self.remaining_cycles -= 1
        return action == self.target_value

#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(Xint, yint)

# Create the scenario instance
problem = ClassifyProblem(Xtr, ytr)
algorithm = XCSAlgorithm()
model = algorithm.new_model(problem)
model.run(problem, learn=True)
print (model)

test_scenario = ClassifyProblem(Xte, yte)
model.run(test_scenario, learn=False)
    #aus diesem output müsste man eigentlich sehen, wie gut die Regeln auf das unbekannte Testset passen, evtl. einfach nur mal die top 5 ausgeben...