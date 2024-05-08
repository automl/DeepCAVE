Displaying Data with Plugins
============================

.. toctree::
   :hidden:
   :maxdepth: 2

   overview
   configurations
   configuration_footprint
   cost_over_time
   configuration_cube
   pareto_front
   parallel_coordinates
   budget_correlation
   importances
   partial_dependencies
   symbolic_explanations


Plugins are used to display data in a specific way. There are plugins to analyse the performance,
the hyperparameters and the budgets. Since AutoML runs are multi-dimensional, the plugins
provide a way to look into specific aspects in more detail.

The goal of the plugins are to provide a way to visualize the data in a way that is easy to
understand. It should give you more insight both data and model-wise and can serve as an additional
sanity check.

User can also add their own plugins to provide deeper insights that suit their needs.


Plugin Structure
----------------

The plugins in DeepCAVE are structured in a specific way. This is done with three blocks:

1. **Input block**: This is used to trigger new calculations. For example, when you change the
  objective in the **Input block**, a new calculation is triggered and cached. The next time you use
  the same input data, no new calculation is needed as the data is already cached. In the end, the
  input data is mapped the calculation, which is referred to as *"raw data"*.

2. **Filter block**: In contrast to the *Input block*, the *Filter block* does not trigger any
  calculation. It acts as a filter over the input data which can be used to influence the display of
  the output.

3. **Output block**: Both the *Input block* and *Filter block* are used to generate information to
  display in the *Output block*. Since the filters are low-cost changes, the output will be updated
  immediately after any filters are changed. The following figure shows the described blocks for the
  cost over time plugin.


.. image:: ../images/plugins/blocks.png


Plugin Types
------------

DeepCAVE has two kinds of plugins, **static** and **dynamic**. While **dynamic** plugins update the
results immediately, **static** plugins are triggered by a `Process` button before displaying their
output.

Most basic plugins should be possible as a **dynamic** plugin, do some quick analysis and then
update the output as soon as it's ready.

In contrast, a **static** plugin might take much longer to generate its output. For example,
the *Hyperparameter Importance* plugin will train many decision trees to calculate it's output and
so the plugins will queue the calculations, viewable in the side bar, and will be cached and
viewable upon completion.

.. note::
    In the current implementation, the cached results are saved to the source directory. You can
    clean the cache directly in the interface.


Custom Plugin
-------------

DeepCAVE was designed so that the plugins require minimal design. We recommend using any of our
provided plugins as a starting point and change it to your needs.

After you have created your plugin, you need to register it in your config file. If you work
on the branch directly, you can adapt ``deepcave/config.py`` to your needs.
We would be very happy to receive pull-requests!

