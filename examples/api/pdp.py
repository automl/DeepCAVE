"""
Partial Dependencies
^^^^^^^^^^^^^^^^^^^^

This example shows how to use the plugin Partial Dependencies.
Note that other plugins use the same interfaces and can be used in the same fashion.
"""

from deepcave.plugins.hyperparameter.pdp import PartialDependencies
from deepcave.runs.converters.deepcave import DeepCAVERun
from pathlib import Path


if __name__ == "__main__":
    # Instantiate the run
    run = DeepCAVERun.from_path(Path("logs/DeepCAVE/minimal/run_2"))
    objective_id = run.get_objective_ids()[0]
    budget_id = run.get_budget_ids()[-1]

    # Instantiate the plugin
    plugin = PartialDependencies()
    inputs = plugin.generate_inputs(
        hyperparameter_name_1="alpha",
        hyperparameter_name_2="beta",
        objective_id=objective_id,
        budget_id=budget_id,
        show_confidence=False,
        show_ice=True,
    )
    # Note: Filter variables are not considered.
    outputs = plugin.generate_outputs(run, inputs)

    # Finally, you can load the figure. Here, the filter variables play a role.
    figure = plugin.load_outputs(run, inputs, outputs)
    figure.write_image("examples/api/pdp.png")
