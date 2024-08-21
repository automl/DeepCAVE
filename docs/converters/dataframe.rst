Pandas DataFrame
----------------

To enable loading files created with additional tools not supported by DeepCAVE, we provide a
converter that can load runs from Pandas DataFrames.
To use this converter, you need to save both the history of trials and the hyperparameter search
space as CSV files and put them in a folder. Then, you can select that folder as your run in
DeepCAVE

We expect the hyperparameter search space to be saved as ``configspace.csv`` in the following format:

+---------------------+-------------+-------+-------------+---------+---------+-------------+---------+---------+---------+-----------------+---------------------+---------------------+------------------+
| name                | type        | log   | lower       | upper   | default | distribution| item_0  | item_1  | ordered | distribution_mu | distribution_sigma  | distribution_alpha  | distribution_beta|
|                     |             |       |             |         |         |             |         |         |         |                 |                     |                     |                  |
+=====================+=============+=======+=============+=========+=========+=============+=========+=========+=========+=================+=====================+=====================+==================+
| alpha               | float       | True  | 1e-08       | 1.0     | 0.01    | normal      |         |         |         | 0.01            | 0.01                |                     |                  |
+---------------------+-------------+-------+-------------+---------+---------+-------------+---------+---------+---------+-----------------+---------------------+---------------------+------------------+
| batch_size          | integer     | True  | 4.0         | 256.0   | 32.0    | uniform     |         |         |         |                 |                     |                     |                  |
+---------------------+-------------+-------+-------------+---------+---------+-------------+---------+---------+---------+-----------------+---------------------+---------------------+------------------+
| depth               | integer     | False | 1.0         | 3.0     | 3.0     | uniform     |         |         |         |                 |                     |                     |                  |
+---------------------+-------------+-------+-------------+---------+---------+-------------+---------+---------+---------+-----------------+---------------------+---------------------+------------------+
| learning_rate_init  | float       | True  | 1e-05       | 1.0     | 0.01    | beta        |         |         |         |                 |                     | 2.0                 | 5.0              |
+---------------------+-------------+-------+-------------+---------+---------+-------------+---------+---------+---------+-----------------+---------------------+---------------------+------------------+
| gradient_clipping   | categorical |       |             |         |         |             | True    | False   | True    |                 |                     |                     |                  |
+---------------------+-------------+-------+-------------+---------+---------+-------------+---------+---------+---------+-----------------+---------------------+---------------------+------------------+

We expect the history of trials to be saved as ``trials.csv`` in the following format:

+-----------+-------------------------+------------+-------+----------------------+-------------------+-------------------------------------+----------+------------+----------+--------+------+------------+
| config_id | alpha                   | batch_size | depth | learning_rate_init   | gradient_clipping | metric:normal [0.0; 1.0] (maximize) | status   | start_time | end_time | budget | seed | additional |
+===========+=========================+============+=======+======================+===================+=====================================+==========+============+==========+========+======+============+
| 0         | 1.3646716470095907e-06  | 50         | 1     | 0.0001145199593038774| true              | 0.2152466367713004                  | success  | 0          | 1        | 1      | -1   |            |
+-----------+-------------------------+------------+-------+----------------------+-------------------+-------------------------------------+----------+------------+----------+--------+------+------------+
| 1         | 0.07654259389007832     | 10         | 2     | 0.006273080002552674 | false             | 0.21973094170403584                 | success  | 1          | 2        | 1      | -1   |            |
+-----------+-------------------------+------------+-------+----------------------+-------------------+-------------------------------------+----------+------------+----------+--------+------+------------+
| 2         | 0.0005027533412617669   | 115        | 1     | 0.00295471450409257  | true              | 0.2017937219730942                  | success  | 2          | 3        | 1      | -1   |            |
+-----------+-------------------------+------------+-------+----------------------+-------------------+-------------------------------------+----------+------------+----------+--------+------+------------+

For some more examples, please have a look at the example runs in `logs/DataFrame`.

Note that the objectives need to be named ``metric:<name> [<lower>; <upper>] (<maximize or minimize>)``, where
``<name>`` is the name of the objective, ``<lower>``, ``<upper>`` are the objective bounds and
``<maximize or minimize>`` is the optimization direction.

The ``status`` column should contain the status of the trial. The following status codes are supported (both upper or lower case):
SUCCESS, TIMEOUT, MEMORYOUT, CRASHED, ABORTED, NOT_EVALUATED, FAILED, PRUNED, UNKNOWN

The ``budget`` column should contain the multi-fidelity budget and can be omitted if not used.
The same holds for the ``seed`` column, containing the trial seed.

.. warning::
    Conditions and forbiddens are not supported in the current version of the Pandas DataFrame converter.
