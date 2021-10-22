from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import random

from deep_cave import app
from deep_cave.plugins.static_plugin import StaticPlugin
from deep_cave.plugins.dynamic_plugin import DynamicPlugin
from deep_cave.utils.logs import get_logger

from deep_cave.evaluators.fanova import fANOVA as _fANOVA


logger = get_logger(__name__)


class MFIS(DynamicPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "mfis"

    @staticmethod
    def name():
        return "Multi-Fidelity Importance Shift"

    @staticmethod
    def position():
        return 3

    @staticmethod
    def category():
        return "Meta-Data Analysis"

    def requirements_met(self, run):
        try:
            run.history[0].additional["cost_history"]
        except:
            return "No cost history found."

        return True

    def get_input_layout(self):
        return [
            dbc.FormGroup([
                dbc.Label("Configurations"),
                dcc.Slider(id=self.register_input(
                    "num_configs", ["min", "max", "marks", "value"]))
            ]),
            dbc.FormGroup([
                dbc.Label("Configuration Selection Seed"),
                dbc.Input(id=self.register_input(
                    "config_selection_seed", "value"), type="number"),
            ]),

            dbc.FormGroup([
                dbc.Label("fANOVA Runs"),
                dbc.Input(id=self.register_input(
                    "num_fanova", "value"), type="number"),
            ]),

            dbc.FormGroup([
                dbc.Label("Trees in fANOVA call"),
                dbc.Input(id=self.register_input(
                    "num_trees", "value"), type="number"),
            ]),

            dbc.FormGroup([
                dbc.Label("Tree Seed"),
                dbc.Input(id=self.register_input(
                    "tree_seed", "value"), type="number"),
            ]),
        ]

    def get_filter_layout(self):
        return [
            dbc.FormGroup([
                dbc.Label("Highest budget (otherwise next budget)"),
                dbc.RadioItems(id=self.register_input(
                    "global", ["options", "value"], filter=True))
            ])
        ]

    def load_inputs(self, run):
        configs = run.get_configs(budget=run.get_highest_budget())

        marks = {}
        for i in range(len(configs)):
            if i % 10 == 0:
                marks[str(i)] = str(i)

        return {
            "num_configs": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": marks,
                "value": len(configs) - 1
            },
            "config_selection_seed": {"value": 0},
            "num_fanova": {"value": 3},
            "num_trees": {"value": 16},
            "tree_seed": {"value": 0},
            "global": {
                "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                "value": 0
            }
        }

    def load_dependency_inputs(self, run, inputs):
        failed = False

        # TODO: Make this smarter here

        try:
            int(inputs["num_fanova"]["value"])
        except:
            inputs["num_fanova"]["value"] = 3
            failed = True

        try:
            int(inputs["num_trees"]["value"])
        except:
            inputs["num_trees"]["value"] = 16
            failed = True

        try:
            int(inputs["tree_seed"]["value"])
        except:
            inputs["tree_seed"]["value"] = 0
            failed = True

        try:
            int(inputs["config_selection_seed"]["value"])
        except:
            inputs["config_selection_seed"]["value"] = 0
            failed = True

        if failed:
            self.update_alert("Only numbers are allowed.", color="danger")

        return inputs

    @staticmethod
    def process(run, inputs):
        # Cost sorted by budget
        costs = {}

        # Shuffle histories
        # random.Random(
        #    int(inputs["config_selection_seed"]["value"])).shuffle(history)

        # Cut histories
        #history = history[:int(inputs["num_configs"]["value"])]
        num_configs = int(inputs["num_configs"]["value"])

        # Get budgets first
        # We use the budgets from the cost history here
        budgets = list(run.history[0].additional["cost_history"].keys())
        highest_budget = budgets[-1]

        history = run.history
        indices = np.arange(len(history))

        # Shuffle
        np.random.seed(int(inputs["config_selection_seed"]["value"]))
        np.random.shuffle(indices)
        history = [history[i] for i in indices]
        history = history[:num_configs]

        for budget in budgets:
            for trial in history:
                if float(highest_budget) != float(trial.budget):
                    continue

                if budget not in costs:
                    costs[budget] = []

                # Get cost from cost history
                cost = trial.additional["cost_history"][budget]

                # And add cost
                costs[budget].append(cost)

        # Do MFIS now
        hp_names = run.configspace.get_hyperparameter_names()

        # Get data for highest budget
        # We have to change Y for different budgets
        X, _ = run.get_encoded_configs(
            budget=run.get_highest_budget(), for_tree=True)

        X = X[indices]
        X = X[:num_configs]

        importance_data = {}
        for budget in budgets:

            # Now get the Y array
            Y = np.array(costs[budget])

            # Same amout of configurations
            assert X.shape[0] == Y.shape[0]

            importance_values = []
            # Run fANOVA x times
            for fanova_run_id in range(int(inputs["num_fanova"]["value"])):
                evaluator = _fANOVA(
                    X, Y,
                    configspace=run.configspace,
                    num_trees=int(inputs["num_trees"]["value"]),
                    seed=int(inputs["tree_seed"]["value"]) + fanova_run_id*100,
                )
                importance_dict = evaluator.quantify_importance(
                    hp_names, depth=1, sorted=False)

                # Make a list of all importance values
                # Hyperparameters are always in the same order
                importance_values.append(
                    [values[1] for values in importance_dict.values()])

            importance_values = np.array(importance_values)
            importance_values = np.mean(importance_values, axis=0)

            # Normalize values s.t. sum equals 1
            importance_values /= np.sum(importance_values)

            importance_data[budget] = list(importance_values)

        return {
            "budgets": budgets,
            "highest_budget": highest_budget,
            "costs": costs,
            "importance_data": importance_data
        }

    def get_output_layout(self):
        return [
            dcc.Graph(self.register_output("graph1", "figure")),
            dcc.Graph(self.register_output("graph2", "figure"))
        ]

    def load_outputs(self, filters, raw_outputs):
        from scipy import stats

        # Collect data
        budgets = raw_outputs["budgets"]
        highest_budget = raw_outputs["highest_budget"]

        budget_correlation = []
        costs = raw_outputs["costs"]

        mfis = []
        importance_data = raw_outputs["importance_data"]

        if filters["global"]["value"] == 1:
            for budget, config_costs in costs.items():
                spearmanrresult = stats.spearmanr(
                    config_costs,
                    costs[highest_budget]
                )
                budget_correlation += [spearmanrresult.correlation]

            # And now we take the highest budget and just calculate abs error (importance shift)
            last_importance_values = importance_data[highest_budget]
            for budget in budgets:
                importance_values = importance_data[budget]

                error = 0
                for current, latest in zip(importance_values, last_importance_values):
                    error += np.abs(current - latest)

                # We also have to divide by the number of configurations
                # because importance data are normalized (sum equals 1)
                # the worst case is that every hyperparameter value differs
                error /= len(importance_values)

                mfis.append(error)

        else:
            for b1, b2 in zip(budgets, budgets[1:]):
                spearmanrresult = stats.spearmanr(
                    costs[b1],
                    costs[b2],
                )
                budget_correlation += [spearmanrresult.correlation]

            # And now we take the highest budget and just calculate abs error (importance shift)
            for b1, b2 in zip(budgets, budgets[1:]):
                i1 = importance_data[b1]
                i2 = importance_data[b2]

                error = 0
                for v1, v2 in zip(i1, i2):
                    error += np.abs(v2 - v1)

                # We also have to divide by the number of configurations
                # because importance data are normalized (sum equals 1)
                # the worst case is that every hyperparameter value differs
                error /= len(i1)

                mfis.append(error)

        trace1 = go.Scatter(
            x=[i for i in range(len(budget_correlation))],
            y=budget_correlation,
            name="Budget Correlation",
            line_shape='hv',
            # hovertext=outputs["additional"]
        )

        trace2 = go.Scatter(
            x=[i for i in range(len(mfis))],
            y=mfis,
            name="MFIS",
            line_shape='hv',
            # hovertext=outputs["additional"]
        )

        layout1 = go.Layout(
            xaxis=dict(
                title='Budgets',
            ),
            yaxis=dict(
                title='Correlation',
            ),
        )

        layout2 = go.Layout(
            xaxis=dict(
                title='Budgets',
            ),
            yaxis=dict(
                title='Importance Shift',
            ),
            yaxis_range=[0, 1],
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        return [fig1, fig2]
