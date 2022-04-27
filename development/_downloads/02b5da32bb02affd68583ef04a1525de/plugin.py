"""
Example Call
^^^^^^^^^^^^

...
"""

from deepcave.plugins.objective.parallel_coordinates import ParallelCoordinates
from deepcave.runs.converters.deepcave import DeepCAVERun


if __name__ == "__main__":
    # Instantiate the run
    run = DeepCAVERun.from_path("examples/record/logs/DeepCAVE/mlp/run_2")
    objective_id = run.get_objective_ids()[0]
    budget_id = run.get_budget_ids()[-1]

    # Instantiate the plugin
    plugin = ParallelCoordinates()
    inputs = plugin.generate_inputs(
        hyperparameter_names=run.configspace.get_hyperparameter_names(),
        objective_id=objective_id,
        budget_id=budget_id,
    )
    # Note: Filter variables are not considered.
    outputs = plugin.generate_outputs(run, inputs)

    # Finally, you can load the figure. Here, the filter variables play a role.
    # Alternatively: Use the matplotlib output (`load_mpl_outputs`) if available.
    figure = plugin.load_outputs(run, inputs, outputs)  # plotly.go figure
    # figure.show()
    # figure.write_image("examples/api/test.png")
