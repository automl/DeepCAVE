Configuration Cube
==================

The configuration cube displays configurations and their score on the objective.
Naturally, the space of configurations may be very high dimensional and so this plugins allows you
to investigate the configurations through 2D or 3D *slices* of the configuration space,
plotting the values of hyperparameters as points in this *slice* of the space.

* **Objective**: Here you can select the objective you wish to analyze for.
  The score for an objective will be communicated through the color of a point with an associated
  heatmap.

* **Budget**: Here you can specify a filter for which budgets you wish to view.
  The *Combined* budget will show all configurations trialed but only show their scores on the
  highest budget if it was evaluated under multiple different budgets.

* **Number of Configurations**: This slider will allow you to select the number of configurations to
  display, where the configurations are ordered by they when they were trialed.
  You can move this along to see how the optimizer chose to evaluate certain hyperparamters over
  time.

* **Hyperparameters**: Here you can select the *slice* with which to look into the configuration
  space. Selecting only one hyperparameter gives you a very basic 1D view, 2 Hyperparameters a 2d
  grid and 3 hyperparamters a 3D cube.
  If multiple configurations map to the same point then the one with the highest score for the
  chosen objective is displayed. This is particular evident in the 1D case where you may have many
  configurations use the same hyperparameter value.


The Configuration Cube is helpful to visually inspect and answer the following questions:

* Does a certain hyperparameter have high influence over the final objectives score when compared to
  another?
* Does the optimizer tend to prefer one value of a hyperparameter over time?
* How do hyperparamters correlate with respect to the given objective


.. image:: ../images/plugins/configuration_cube.png
