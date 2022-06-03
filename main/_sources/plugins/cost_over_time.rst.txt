Cost Over Time
==============

One goal of using AutoML is to gain high-peak performance. However, you also want to see
how the objectives change over time. To see the change based on the time or number of
configurations, this plugin visualize it for you.

Since multiple runs are supported, you directly see which run performs best to which time.
If you decide to display groups (which are combined runs), you will see the mean and standard
deviation too.

.. note:: 
    The configuration spaces of the selected runs have to be equal. Otherwise, a good comparison
    is not possible.

This plugin is capable of answering following questions:

* Does the optimizer converge?
* How performs optimizer A in comparison to optimizer B? Is optimizer A better than optimizer B?
* How long does the optimizer need to reach a certain objective value?


Inputs and Filters
------------------
The objective is used to compare the runs on the y-axis. The option "x-axis" is a
convenient way to influence the x-axis to your needs. If you are viewing a run which takes long,
then the logarithmic option would be best for you. If you want to know how many configurations the
optimizer needed to achieve a specific result, then "evaluated configurations" would be the choice
for you.

Aditionally, you can specify whether you would like to see runs and/or groups or not.


Interpretation
--------------

As mentioned above, the resulting graph shows the objective values on the y-axis and the
time on the x-axis. Based on the lines, you can see when a run/group is performing better/worse
to the others and to which time. This graph is perfect to determine which optimizer converges
faster and/or if the optimizer converges at all.

In contrast to the runs, groups show the mean and standard deviation of all included runs of
the group. Use this feature to see how much spread between multiple runs exists.


.. image:: ../images/plugins/cost_over_time.png
