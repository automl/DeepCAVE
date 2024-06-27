Configuration Footprint
=======================

Configuration footprint is useful for exploring how much is known about your configuration space by
the optimizer and view its preferences during the run. These can be generated and will be cached
when hitting the `Process` button.

This plugin is capable of answering following questions:

* Was the configuration space well-covered by the optimizer?
* Can I stop the optimization process or do I need to invest more computational resources?
* Which hyperparameter values are favorable?

First, let's briefly mention the various kind of configurations we are concerned with here.

* **Incumbent**: This is the best configuration for a given objective, like *cost* or *time*,
  shown as a red triangle.
* **Evaluated Configuration**: These are configurations evaluated by the optimizer and we know the
  true objective value, shown with *orange x*'s.
* **Unevaluated Configuration**: These are configurations sampled by the optimizer but have no been
  evaluated, shown with *purple x*'s.
* **Border Configuration**: These are configurations that exist at the edge of the configuration
  space at the min and max values for scalars.

Using the **Evaluated Configuration** and the **Incumbent** for each objective type and under each
budget, we have some knowledge about what kind of score we'll get around those points, and we can
try to infer for all other points in the configuration space.

The next point to address is how we represent a high dimensional Configuration in something we can
plot in 2D. This is done using a dimensionality reduction algorithm
`MDS <https://en.wikipedia.org/wiki/Multidimensional_scaling>`_ which attempts to preserve distances
as well as possible when converting from the high dimensional spaceto the lower dimensional one.
Of course this is not perfect but it can still provide a lot of insight.

Now we can begin to talk about the plots. These two plots share the same axis and and points will be
plotted in the same co-ordinates so swapping between them will be most beneficial to get a firm
grasp on your configuration footprint.


Performance plot
----------------
Perhaps the most useful plot of the two is to see how much we know about which configuration will
achieve which **objective** score. To view this, mousing over the **Incumbent** will tell you the
best configuration found for the given *objective* and *budget*. For non-deterministic runs (i.e.
multiple seeds evaluated per configuration), only configurations evaluated on the maximum number of
seeds are considered to choose the best configuration from. The **Evaluated Configuration**
points will have a true objective score, all be it possibly noisy if the objective is noisy.
Using these two kind of points, we can try to infer what the performance for the rest of the
configuration space will be.

It's important to note that the background colour, represnting the objective is a best estimate
given what we already know and does not indicate the true objective value in regions where there
are no configurations evaluated. We can increase the resolution using the *Details* option, which
influences the gridsize when generating the plot, at the cost of increased compute time to generate
it.


Coverage plot
-------------
Here you'll get a nice overview of how much your configuration space is sampled and under what
regions it is most evaluated. This is where the **Border Configurations** can be quite informative
to view. Ideally you will see these **Border Configurations** at the borders of the different blobs,
indicating the bounds of the 2d representation of your configuration space. There will likely be
small clusters of **Evaluated Points** where the optimizer focused on finding good configurations
as well as some points scattered through the valid configuration regions, to get a good
understanding of the scores for the various objectives.


.. image:: ../images/plugins/configuration_footprint.png
