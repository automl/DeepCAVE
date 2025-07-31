"""
Parallel Coordinates
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Parallel Coordinates.
Note that other plugins use the same interfaces and can be used in the same fashion.
To use the API commands in this example, we assume the run data to be already present in the mentioned folder. Please ensure that you have the run data located at the corresponding path.
"""

from deepcave.plugins.hyperparameter.parallel_coordinates import ParallelCoordinates
from deepcave.runs.converters.deepcave import DeepCAVERun
from pathlib import Path


if __name__ == "__main__":
    # Instantiate the run
    run = DeepCAVERun.from_path(Path("logs/DeepCAVE/minimal/run_2"))
    objective_id = run.get_objective_ids()[0]
    budget_id = run.get_budget_ids()[-1]

    # Instantiate the plugin
    plugin = ParallelCoordinates()
    inputs = plugin.generate_inputs(
        hyperparameter_names=list(run.configspace.keys()),
        objective_id=objective_id,
        budget_id=budget_id,
        show_important_only=True,
        show_unsuccessful=False,
        hide_hps=False,
        n_hps=3,
    )
    # Note: Filter variables are not considered.
    outputs = plugin.generate_outputs(run, inputs)

    # Finally, you can load the figure. Here, the filter variables play a role.
    figure = plugin.load_outputs(run, inputs, outputs)
    figure.write_image("examples/api/parallel_coordinates.png", scale=2.0)
