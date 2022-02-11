import dash_bootstrap_components as dbc
import pandas as pd
from dash import html

from deepcave import run_handler
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import AbstractRun, Status
from deepcave.runs.grouped_run import NotMergeableError


class Overview(DynamicPlugin):
    id = "overview"
    name = "Overview"
    icon = "fas fa-search"

    activate_run_selection = True

    @staticmethod
    def process(run, inputs):
        # Meta
        meta = {"Attribute": [], "Value": []}
        for k, v in run.get_meta().items():
            if k == "objectives":
                continue

            if isinstance(v, list):
                v = ", ".join(str(_v) for _v in v)

            meta["Attribute"].append(k)
            meta["Value"].append(str(v))

        # Objectives
        objectives = {"Name": [], "Lower Bound": [], "Upper Bound": []}
        for objective in run.get_objectives():
            objectives["Name"].append(objective["name"])
            objectives["Lower Bound"].append(objective["lower"])
            objectives["Upper Bound"].append(objective["upper"])

        budgets = run.get_budgets()
        stats = {}
        # Statistics
        for budget in budgets:

            if budget not in stats:
                stats[budget] = {}

                for s in Status:
                    stats[budget][s.name] = 0

        for trial in run.get_trials():
            stats[trial.budget][trial.status.name] += 1

        statistics = {"Budget": budgets, "Total": []}
        for s in Status:
            statistics[s.name] = []

            for budget in budgets:
                count = stats[budget][s.name]

                # Calculate max entries
                count_all = 0
                for _count in stats[budget].values():
                    count_all += _count

                if len(statistics["Total"]) < len(budgets):
                    statistics["Total"].append(count_all)

                percentage = int(count / count_all * 100)

                string = f"{count} ({percentage}%)"
                statistics[s.name].append(string)

        return {
            "meta": meta,
            "objectives": objectives,
            "statistics": statistics,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            html.H3("Meta"),
            html.Div(id=register("meta", "children")),
            html.Hr(),
            html.H3("Objectives"),
            html.Div(id=register("objectives", "children")),
            html.Hr(),
            html.H3("Statistics"),
            html.Div(id=register("statistics", "children")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):
        run = run_handler.from_run_id(inputs["run_name"]["value"])
        outputs = outputs[run.name]

        def create_table(output):
            return dbc.Table.from_dataframe(
                pd.DataFrame(output), striped=True, bordered=True
            )

        return [
            create_table(outputs["meta"]),
            create_table(outputs["objectives"]),
            create_table(outputs["statistics"]),
        ]
