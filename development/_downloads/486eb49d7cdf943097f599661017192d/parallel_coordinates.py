"""
Parallel Coordinates
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Parallel Coordinates.
Note that other plugins use the same interfaces and can be used in the same fashion.
"""

from deepcave.plugins.objective.parallel_coordinates import ParallelCoordinates
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
        hyperparameter_names=run.configspace.get_hyperparameter_names(),
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
    # Alternatively: Use the matplotlib output (`load_mpl_outputs`) if available.
    figure = plugin.load_outputs(run, inputs, outputs)  # plotly.go figure
    figure.write_image("examples/api/parallel_coordinates.png", scale=2.)
    # figure.show()
