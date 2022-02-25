Converters
==========

DeepCAVE uses :ref:`Run<deepcave.runs.run>` objects to interpret the input data. Converters make sure that folders
are put into the right format so that the files can be interpreted correctly.

Currently, DeepCAVE supports the following converters:

- DeepCAVE (native)
- SMAC (v1.1.1)
- BOHB


Custom Converter
^^^^^^^^^^^^^^^^

To support a new converter, you need to create a new class that inherits from :ref:`Run<deepcave.runs.run>`.
Two methods are needed:
- `hash`: This method should return a unique hash for the run (e.g. hashing the trial history).
- `from_path`: This method should return a :ref:`Run<deepcave.runs.run>` object from a path. Inside this method, you have
to provide a configspace, meta data, :term:`objectives <Objective>`, :term:`budgets <Budget>` and add the trials to the new created run object.

After you have created a new converter, you need to add it to ``AVAILABLE_CONVERTERS`` in
:ref:`deepcave/config.py<deepcave.config>`. The next time you select a run, DeepCAVE will automatically use this converter as long
as the ``from_path`` method did not fail.

