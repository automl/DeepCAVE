Getting Started
===============

After you have followed the installation steps, you can start DeepCAVE via following command:

.. code:: bash

    $ deepcave --start


If you did not change the address or ports, you can access the interactive dashboard
via `http://127.0.0.1:8050`. In the dashboard, you will be directed to the main page, in which you
can select the runs you want to analyze. Please have a look at the following image
to see the dashboard in action:

.. image:: images/plugins/interface.png


DeepCAVE uses runs to interprete data. A run is a collection of trials (a configuration with associated costs).
Once valid runs (see :ref:`converter<Converters>`) are found, you can select them to analyze them further.
After selecting runs, you can also add them to groups to analyze combined runs easily.
