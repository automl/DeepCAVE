"""
Importances
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Importances.
Note that other plugins use the same interfaces and can be used in the same fashion.
"""

from pathlib import Path

from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.runs.converters.deepcave import DeepCAVERun

if __name__ == "__main__":
    # Instantiate the run
    run = DeepCAVERun.from_path(Path("logs/DeepCAVE/minimal/run_2"))

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
    # Alternatively: Use the matplotlib output (`load_mpl_outputs`) if available.
    figure = plugin.load_outputs(run, inputs, outputs)  # plotly.go figure
    figure.write_image("examples/api/importances.png", scale=2.0)
    # figure.show()
