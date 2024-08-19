Custom
------

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
