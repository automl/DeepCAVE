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
- AMLTK


.. note::
    DeepCAVE observes optimizer data on the file system and therefore allows for monitoring of both
    finished processes and running processes that regularly write new results to disk as long as a
    converter is available.

AMLTK Converter
----------------

To be able to load your AMLTK run into DeepCave, there are a few points to
consider when running AMLTK:

1. Save the ``trial_history`` as a CSV file:

   .. code-block:: python

      history_df = trial_history.df()
      bucket.store({"history.csv": history_df})

2. Save the configuration space as a ConfigSpace JSON file:

   .. code-block:: python

      from ConfigSpace.read_and_write import json as cs_json
      space = pipeline.search_space(parser="configspace")
      bucket.store({"configspace.json": cs_json.write(space)})

3. Define the start and end times for your trials:

   - DeepCAVE needs to know which columns to use as the trial start end end times.
     Ensure the start and end times are named in the format ``"deepcave:time:start"`` and
     ``"deepcave:time:end"`` in the ``history.csv``.
   - You can define these times during the AMLTK run setup. For example, assuming you have a
     data-loading and a scoring step and tracked their times via the profiler, to use the start time
     from data-loading and the end time of scoring as DeepCAVE start end end time, you can add the
     following lines before calling ``bucket.store()`` in step 1:

     .. code-block:: python

        history_df["deepcave:time:start"] = history_df["profile:data-loading:time:start"]
        history_df["deepcave:time:end"] = history_df["profile:scoring:time:start"]

   - Alternatively, you can manually add the DeepCave time columns to ``history.csv`` after the
     AMLTK run has finished.


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

