Converters
==========

DeepCAVE uses :ref:`Run<deepcave.runs.run>` objects to interpret the input data. Converters make
sure that folders are put into the right format so that the files can be interpreted correctly.

Currently, DeepCAVE supports the following converters:

- DeepCAVE (native)
- SMAC (v1.4)
- SMAC (v2.0.0)
- BOHB
- Auto-Sklearn
- Auto-PyTorch


.. note::
    DeepCAVE observes optimizer data on the file system and therefore allows for monitoring of both
    finished processes and running processes that regularly write new results to disk as long as a
    converter is available.


Custom Converter
----------------

To support a new converter, you need to create a new class that inherits from
:ref:`Run<deepcave.runs.run>`. Three methods are needed:

- **hash**: Returns a unique hash for the run (e.g. hashing the trial history).
- **latest_change**: Returns when the latest change was.
- **from_path**: Returns a :ref:`Run<deepcave.runs.run>` object from a path. Inside this method,
  you have to provide a configspace, meta data, :term:`objectives <Objective>`,
  :term:`budgets <Budget>` and add the trials to the new created run object.

After you have created a new converter, you need to add it to ``AVAILABLE_CONVERTERS`` in your
config (or if you work directly on the branch change ``deepcave/config.py`` to your needs).
The next time you select a run, DeepCAVE will automatically use this converter as long as the
``from_path`` method did not fail.

