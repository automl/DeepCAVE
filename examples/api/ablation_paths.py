"""
Ablation Paths
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Ablation Paths.
Note that other plugins use the same interfaces and can be used in the same fashion.
To use the API commands the according record example needs to be run in order create the run to analyze.
"""

from deepcave.plugins.hyperparameter.ablation_paths import AblationPaths
from deepcave.runs.converters.deepcave import DeepCAVERun
from pathlib import Path


if __name__ == "__main__":
    # Instantiate the run
    run = DeepCAVERun.from_path(Path("logs/DeepCAVE/minimal/run_1"))

    objective_id1 = run.get_objective_ids()[0]
    objective_id2 = None  # replace with run.get_objective_ids()[1] for multi-objective importance
    budget_id = run.get_budget_ids()[0]

    # Instantiate the plugin
    plugin = AblationPaths()
    inputs = plugin.generate_inputs(
        objective_id1=objective_id1,
        objective_id2=objective_id2,
        budget_id=budget_id,
        n_hps=100,
        n_trees=100,
        show_confidence=False,
    )
    # Note: Filter variables are not considered.
    outputs = plugin.generate_outputs(run, inputs)

    # Finally, you can load the figure. Here, the filter variables play a role.
    figure1, figure2 = plugin.load_outputs(run, inputs, outputs)
    figure1.write_image("examples/api/ablation_paths_performance.png", scale=2.0)
    if not objective_id2:
        figure2.write_image("examples/api/ablation_paths_improvement.png", scale=2.0)
