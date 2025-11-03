"""
# HyperSHAP


## Classes
    - HyperSHAP:
"""
from typing import Callable, List, Dict, Any
from deepcave.runs import AbstractRun
from plotly.graph_objs import go
from deepcave.plugins.static import StaticPlugin

class HyperSHAP(StaticPlugin):
    id = "hypershap"
    name = "HyperSHAP"
    icon = "My first plugin icon"
    @staticmethod
    def get_input_layout(register: Callable) -> List:
        """
        Define the input block of the plugin.
        """
    @staticmethod
    def get_filter_layout(register: Callable) -> List:
        """
        Define the filter block of the plugin.
        """

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.
        """
    def load_dependency_inputs(self, run: AbstractRun, previous_inputs: Dict[str, Any], inputs: Dict[str, Any],) -> Dict[str, Any]:
        """
        Works like 'load_inputs' but called after inputs have changed.
        """
    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process your input data and return raw data to be used in the output layout.
        """
    @staticmethod
    def get_output_layout(register: Callable):
        """
        Define the output block of the plugin.
        """
    @staticmethod
    def load_outputs(runs, inputs, outputs) -> go.Figure:
        """
        Load the raw output data for the plugin and create a figure to be shown in the output block.
        """