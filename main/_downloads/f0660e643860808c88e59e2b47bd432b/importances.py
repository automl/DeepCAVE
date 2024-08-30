"""
Importances
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Importances.
Note that other plugins use the same interfaces and can be used in the same fashion.
"""

from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.runs.converters.deepcave import DeepCAVERun
from pathlib import Path


if __name__ == "__main__":
    # Instantiate the run
    run = DeepCAVERun.from_path(Path("logs/DeepCAVE/minimal/run_2"))

    objective_id1 = run.get_objective_ids()[0]
    objective_id2 = None  # replace with run.get_objective_ids()[1] for multi-objective importance
    budget_ids = run.get_budget_ids()

    # Instantiate the plugin
    plugin = Importances()
    inputs = plugin.generate_inputs(
        hyperparameter_names=list(run.configspace.keys()),
        objective_id1=objective_id1,
        objective_id2=objective_id2,
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
