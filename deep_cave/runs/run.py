import ConfigSpace
from smac.runhistory.runhistory import RunHistory


class Run:
    def __init__(self, meta: dict, runhistory: RunHistory, configspace: ConfigSpace):

        """
        meta: start_time, end_time, duration, 
        """

        self.meta = meta
        self.rh = runhistory
        self.cs = configspace
    
    def get_meta(self):
        return self.meta
    
    def get_runhistory(self):
        return self.rh

    def get_configspace(self):
        return self.cs

    def get_fidelities(self):
        budgets = []

        runkeys = self.rh.data
        for runkey in runkeys:
            budget = runkey.budget

            if budget not in budgets:
                budgets.append(budget)

        return budgets
