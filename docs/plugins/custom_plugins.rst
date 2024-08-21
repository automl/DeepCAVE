.. _custom-plugin:

================
Adding a Plugin
================

Introduction
------------

This tutorial will guide you through the steps to add a new plugin in DeepCAVE.
We will create a simple plugin that demonstrates the basic structure and functionality.
By the end of this tutorial, you will hopefully have a better understanding of how to create and use plugins in DeepCAVE.

Step 1: Create the plugin class
-------------------------------

1. Start by creating a new Python file in the `deepcave/plugins` directory of your DeepCAVE installation.
   For example, you can create a file named `my_first_plugin.py`.

2. Define the basic structure of the plugin. You can start by using the following code:

    .. code-block:: python

        class MyFirstPlugin(PluginType):
            id = "my_first_plugin"
            name = "My first plugin"
            icon = "My first plugin icon"
            @staticmethod
            def get_input_layout(register: Callable) -> List[]:
                """
                Define the input block of the plugin.
                """
            @staticmethod
            def get_filter_layout(register: Callable) -> List[]:
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

Understand the structure of the layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To understand where each part of the layout belongs to, please refer to the following depiction:

.. image:: ../images/plugins/blocks.png


Step 2: Register your plugin
-----------------------------

To fully integrate your plugin, you need to add it to the list of available plugins per category in the `deepcave/config.py` file.
The plugin needs to be imported and then added to the plugin dictionary.
In addition, please add a short description to the `__init__.py` file of the corresponding category in the `deepcave/plugins` directory.

Step 3: Decide on your plugin type
----------------------------------

Your plugin can either be dynamic or static. You need to substitute `PluginType` in the example code with either
`DynamicPlugin` or `StaticPlugin`.

If your plugin needs more processing time or computing power, we recommend you choose `StaticPlugin` as your plugin type,
as this way it will not immediately start processing once you select it.
Otherwise you can choose the `DynamicPlugin`, which will execute directly when you select the plugin
(see :ref:`static plugin <plugin_types>` for more information).

Step 4: Adjust the functions of your plugin
-------------------------------------------

In Step 1 you saw how the basic structure of the plugin should look like.
Please refer to the :ref:`api references<api_references>` documentation, so you can fit these functions to your specific needs.

Note that you might want to add a `check_runs_compatability` functionality if you want to analyze more than one run.

Step 5: Fit the plugin to your needs
------------------------------------

Adjustment 1: Activate run selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can add a select bar at the top of your plugin, where the run(s) to analyze can be manually selected.
To do this, simply add `activate_run_selection` to your class variables and set it to `True`.

Adjustment 2: Icon
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can choose your own icon from the `Font Awesome <https://fontawesome.com/start>`_ library.
Please note that at this moment icons from the newer versions may not be available.

    .. code-block:: python

        # Example icon
        icon = "fas fa-chart-line"

Step 6: Document your plugin
-----------------------------

To document your plugin, create a corresponding `.rst` file under `docs/plugins` and add it to
the `docs/plugins/index.py` file.
Additionally, define a class variable in your `my_first_plugin.py` with the path to your documentation, called `help`.

.. code-block:: python

    # Example of help variable
    help = "docs/plugins/my_first_plugin.rst"

Step 7: Congratulate yourself
-----------------------------

Congratulations, you just wrote your first DeepCAVE plugin!
If you have any questions, please do not hesitate to contact us.
