Displaying Data with Plugins
============================

.. toctree::
   :hidden:
   :maxdepth: 2

   overview
   configurations
   configuration_footprint
   cost_over_time
   pareto_front
   budget_correlation
   importances
   ablation_paths
   configuration_cube
   parallel_coordinates
   partial_dependencies
   symbolic_explanations
   custom_plugins


Plugins are used to display data in a specific way. There are plugins to analyse the performance,
the hyperparameters, and the budgets. Since AutoML runs are multi-dimensional, the plugins
provide a way to look into specific aspects in more detail.

The goal of the plugins is to provide a way to visualize the data in a way that is easy to
understand. It should give you more insight both regarding the final hyperparameter configuration
and the HPO process and can serve as an additional sanity check.

Plugin Structure
----------------

The plugins in DeepCAVE are structured in a specific way. This is done with three blocks:

1. **Input block**: This is used to trigger new calculations. For example, when you change the
   objective in the *Input block*, a new calculation is triggered and cached. The next time you use
   the same input data, no new calculation is needed as the data is already cached.

2. **Filter block**: In contrast to the *Input block*, the *Filter block* does not trigger any
   calculation. It acts as a filter over the input data which can be used to influence the display of
   the output.

3. **Output block**: Both the *Input block* and *Filter block* are used to generate information to
   display in the *Output block*. Since the filters are low-cost changes, the output will be updated
   immediately after any filters are changed. The following figure shows the described blocks for the
   Cost over Time plugin.


.. image:: ../images/plugins/blocks.png


.. _plugin_types:

Plugin Types
------------

DeepCAVE offers two kinds of plugins: **static** and **dynamic**. While **dynamic** plugins update the
results immediately, **static** plugins are triggered by a `Process` button before displaying their
output.

**Dynamic** plugins should be used if the plugin is supposed to do some quick analysis and then
update the output as soon as the analysis is ready.

In contrast, a **static** plugin might take much longer to generate its output. For example,
the *Hyperparameter Importance* plugin will train many decision trees to calculate its output.
Therefore, these kind of plugins queue their calculations, viewable in the side bar. Upon completion,
the results will be shown and cached. The next time you open the plugin with the corresponding run,
the cached results will be displayed immediately.

.. note::
    In the current implementation, the cached results are saved to the source directory. You can
    clean the cache directly in the interface, on the main page.


Adding a Plugin
--------------

If you would like to add your own plugin, please have a look at our :ref:`tutorial on how to add
a plugin <custom-plugin>` and use any of our provided plugins as a starting point.


We would be very happy to receive pull-requests!

