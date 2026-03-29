#  noqa: D400
"""
# HyperSHAP

## Classes
    - HyperSHAP:
"""
from typing import Any, Callable, Dict, List

import ast
import math

import dash_bootstrap_components as dbc
import matplotlib
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import config
from deepcave.evaluators.hypershap import HyperSHAP_Eval as Evaluator
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button

matplotlib.use("Agg")


class Tunability(StaticPlugin):
    """Provide a Plugin for the Tunability analysis of HyperSHAP."""

    id = "tunability"
    name = "Tunability"
    icon = "fas fa-binoculars"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List:
        """Define the input block of the plugin."""
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Tunability"),
                            dbc.Select(
                                id=register("tunability", ["value", "options"], type=str),
                                placeholder="Select tunability ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[dbc.Row]:
        """
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dbc.Row]
            Layout for the filter block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Label("Budgets"),
                                    help_button(
                                        "Budget refers to the multi-fidelity budget. "
                                        "The hyperparameters are sorted by the highest budget."
                                    ),
                                    dbc.RadioItems(
                                        id=register("budget_id", ["value", "options"]), inline=True
                                    ),
                                ]
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'."""
        return {
            "tunability": {"options": get_select_options(labels=["Tunability", "Mistunability"])},
            "objective": {"options": get_select_options()},
            "budget_id": {"options": get_checklist_options(), "value": None},
        }

    def load_dependency_inputs(self, run, _, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Work like 'load_inputs' but called after inputs have changed.

        It is restricted to three Hyperparameters.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Current content of the inputs.

        Returns
        -------
        Dict[str, Any]
           The dictionary with the changes.
        """
        # Prepare objectives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)
        objective_value = inputs["objective_id"]["value"]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_id"]["value"]

        if run is not None:
            if budget_value is None:
                budget_value = budget_ids[-1]

        tune_value = inputs["tunability"]["value"]

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "tunability": {
                "options": get_select_options(labels=["Tunability", "Mistunability"]),
                "value": tune_value,
            },
            "budget_id": {
                "options": budget_options,
                "value": budget_value,
            },
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process your input data and return raw data to be used in the output layout."""
        eval = Evaluator(run)

        eval.hype_tune(
            tunability=inputs["tunability"],
            objective_id=inputs["objective_id"],
            budget_id=inputs["budget_id"],
        )

        return {"inputs": eval.get_tunability()}

    @staticmethod
    def get_output_layout(register: Callable) -> Any:
        """Define the output block of the plugin."""
        return [
            dcc.Graph(
                register("perf_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            )
        ]

    @staticmethod
    def load_outputs(runs, inputs, outputs) -> go.Figure:  # type: ignore
        """Read the raw data and prepare it for the layout."""
        iv_values = outputs["inputs"]

        restored_interactions = {}
        for key_str, value in iv_values["interactions"].items():
            if key_str == "()":
                restored_interactions[()] = value
            else:
                # ast.literal_eval safely evaluates strings into tuples (e.g., "(0, 1)" -> (0, 1))
                restored_interactions[ast.literal_eval(key_str)] = value

        feature_names = ["a", "b", "c"]
        n_nodes = len(feature_names)
        data_dict = restored_interactions
        # --- Configuration & Styling ---
        # SHAP standard colors
        COLOR_POS = "rgba(255, 13, 87, 1.0)"  # Red for positive
        COLOR_NEG = "rgba(30, 136, 229, 1.0)"  # Blue for negative
        HALO_POS = "rgba(255, 13, 87, 0.3)"  # Lighter red halo
        HALO_NEG = "rgba(30, 136, 229, 0.3)"  # Lighter blue halo

        MAX_NODE_SIZE = 80  # Max pixel size for the largest node
        MAX_EDGE_WIDTH = 25  # Max pixel thickness for the strongest line

        # --- AUTOMATIC SCALING MAGIC ---
        # Find the maximum absolute value in your data so sizes never explode or vanish
        all_values = [abs(v) for k, v in data_dict.items() if len(k) > 0]
        max_val = max(all_values) if all_values else 1
        if max_val == 0:
            max_val = 1  # Prevent division by zero

        fig = go.Figure()

        # Calculate coordinates (arrange nodes in a circle)
        positions = {}
        for i in range(n_nodes):
            angle = 2 * math.pi * i / n_nodes + (math.pi / 2)
            positions[i] = (math.cos(angle), math.sin(angle))

        # 2. Draw Edges (2nd-Order Interactions)
        for key, val in data_dict.items():
            if len(key) == 2:
                u, v = key  # type: ignore
                x0, y0 = positions[u]  # type: ignore
                x1, y1 = positions[v]  # type: ignore

                color = COLOR_POS if val > 0 else COLOR_NEG

                # Scale the line thickness relative to the maximum value
                normalized_width = (abs(val) / max_val) * MAX_EDGE_WIDTH

                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=normalized_width, color=color),
                        hoverinfo="text",
                        text=f"{feature_names[u]}"  # type: ignore
                        f"& {feature_names[v]} interaction: {val:.4f}",  # type: ignore
                        showlegend=False,
                    )
                )

        # 3. Process Nodes (1st-Order Effects)
        node_x, node_y = [], []
        node_halo_sizes, node_colors, halo_colors = [], [], []
        hover_texts = []

        for i in range(n_nodes):
            val = data_dict.get((i,), 0)  # type: ignore
            x, y = positions[i]
            node_x.append(x)
            node_y.append(y)

            node_colors.append(COLOR_POS if val >= 0 else COLOR_NEG)
            halo_colors.append(HALO_POS if val >= 0 else HALO_NEG)

            # Scale the colored ring size relative to the maximum value
            normalized_size = (abs(val) / max_val) * MAX_NODE_SIZE
            node_halo_sizes.append(normalized_size)

            hover_texts.append(f"{feature_names[i]} main effect: {val:.4f}")

        # 4. Draw the Nodes Layer by Layer

        # Layer A: The transparent outer halo
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(size=[s * 1.3 for s in node_halo_sizes], color=halo_colors),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Layer B: The solid colored circle
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(size=node_halo_sizes, color=node_colors),
                hoverinfo="text",
                text=hover_texts,
                showlegend=False,
            )
        )

        # Layer C: The inner white circle with the text (Fixed size so it never vanishes)
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=40, color="white", line=dict(color="black", width=1.5)),
                text=feature_names,
                textposition="middle center",
                textfont=dict(size=14, color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # 5. Clean up the canvas

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=30, b=0, l=0, r=0),
            autosize=True,
        )

        return fig
