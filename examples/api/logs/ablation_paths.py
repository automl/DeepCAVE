"""
Ablation Paths
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Ablation Paths.
Note that other plugins use the same interfaces and can be used in the same fashion.
"""

from deepcave.plugins.hyperparameter.ablation_paths import AblationPaths
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from pathlib import Path


if __name__ == "__main__":
    # Instantiate the run
    run = SMAC3v1Run.from_path(Path("logs/SMAC3v1/outlier-detection/DEHB-pendigits-015-0-0"))

    objective_id = run.get_objective_ids()[0]
    budget_id = run.get_budget_ids()[0]

    # Instantiate the plugin
    plugin = AblationPaths()
    inputs = plugin.generate_inputs(
        objective_id=objective_id,
        budget_id=budget_id,
        n_hps=40,
        n_trees=100,
    )
    # Note: Filter variables are not considered.
    outputs = plugin.generate_outputs(run, inputs)

    # Finally, you can load the figure. Here, the filter variables play a role.
    # Alternatively: Use the matplotlib output (`load_mpl_outputs`) if available.
    figure = plugin.load_outputs(run, inputs, outputs)  # plotly.go figure
    figure.write_image("examples/api/ablation_paths.png", scale=2.0)
    # figure.show()
