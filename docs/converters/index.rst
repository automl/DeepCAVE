Runs and Converters
===================

.. toctree::
   :hidden:
   :maxdepth: 2

   smac
   amltk
   optuna
   bohb
   deepcave
   dataframe
   custom_converters


The Run Object
--------------

DeepCAVE utilizes :ref:`Run<deepcave.runs.run>` objects as a fundamental unit for data
interpretation.
A run represents a hyperparameter optimization process, encompassing a collection of
trials, each corresponding to a specific hyperparameter configuration with its associated
objective value, budget, and seed.

Converters
----------

Converters are used to access the optimizer data stored on the file system and
transform it into run objects.

Currently, DeepCAVE offers the following converters:

- SMAC (v1.4)
- SMAC (v2.0.0)
- AMLTK
- Optuna
- BOHB
- DeepCAVE (native)
- Pandas DataFrame

In the `logs` directory, you can find example runs for each of the converters.


.. note::
    DeepCAVE observes optimizer data on the file system and therefore allows for monitoring of both
    finished processes and running processes that regularly write new results to disk.

Adding a Converter
------------------

If you would like to add your own converter, please have a look at our :ref:`tutorial on how to add
a converter <custom-converter>` and use any of our provided converters as a starting point.