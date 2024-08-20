"""
Importances
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Importances.
Note that other plugins use the same interfaces and can be used in the same fashion.
"""

from pathlib import Path

from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.runs.converters.dataframe import DataFrameRun

if __name__ == "__main__":
    # Instantiate the run
    run = DataFrameRun.from_path(Path("logs/DataFrame/run_1"))

    objective_id = run.get_objective_ids()[0]
    budget_ids = run.get_budget_ids()

    # Instantiate the plugin
    plugin = Importances()
    inputs = plugin.generate_inputs(
        hyperparameter_names=list(run.configspace.keys()),
        objective_id=objective_id,
        budget_ids=budget_ids,
        method="global",
        n_hps=3,
        n_trees=10,
    )
    # Note: Filter variables are not considered.
    outputs = plugin.generate_outputs(run, inputs)

    # Finally, you can load the figure. Here, the filter variables play a role.
    figure = plugin.load_outputs(run, inputs, outputs)
    figure.write_image("examples/api/importances.png", scale=2.0)
