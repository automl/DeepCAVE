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
- Optuna


.. note::
    DeepCAVE observes optimizer data on the file system and therefore allows for monitoring of both
    finished processes and running processes that regularly write new results to disk as long as a
    converter is available.

AMLTK Converter
----------------

To be able to load your AMLTK run into DeepCave, there are a few points to
consider when running AMLTK:

1. Save the ``trial_history`` as a Parquet file:

   .. code-block:: python

      history_df = trial_history.df()
      bucket.store({"history.parquet": history_df})

2. Save the configuration space as a ConfigSpace JSON file:

   .. code-block:: python

      space = pipeline.search_space(parser="configspace")
      bucket.store({"configspace.json": space.to_serialized_dict()})

3. Define the start and end times for your trials:

   - DeepCAVE needs to know which columns to use as the trial start end end times.
     Ensure the start and end times are named in the format ``"deepcave:time:start"`` and
     ``"deepcave:time:end"`` in the ``history.parquet``.
   - You can define these times during the AMLTK run setup. For example, assuming you have a
     data-loading and a scoring step and tracked their times via the profiler, to use the start time
     from data-loading and the end time of scoring as DeepCAVE start end end time, you can add the
     following lines before calling ``bucket.store()`` in step 1:

     .. code-block:: python

        history_df["deepcave:time:start"] = history_df["profile:data-loading:time:start"]
        history_df["deepcave:time:end"] = history_df["profile:scoring:time:start"]

   - Alternatively, you can still manually add the DeepCave time columns to ``history.parquet``
     after the AMLTK run has finished by loading it into a Pandas Dataframe, manipulating it, and
     writing it back to the ``history.parquet`` file.

Optuna Converter
----------------

To be able to load your Optuna run into DeepCave, there are a few points to
consider when running Optuna:

1. Save the Optuna ``study`` object as a Pickle file:

   .. code-block:: python

        import pickle
        with open("<path-to-save-study>/optuna_study.pkl", "wb") as f:
            pickle.dump(study, f)

   Replace ``<path-to-save-study>`` with the directory where you want to store the Pickle file.

2. Optional: If you want to display the names of your objectives in DeepCAVE, you can set metric names in Optuna using the following command:

   .. code-block:: python

        study.set_metric_names(["Accuracy", "Flops"])

   This step is optional but recommended for better visualization and understanding of your study's objectives in DeepCAVE.

.. warning::
    Loading Optuna runs with conditional search spaces or dynamic hyperparameter value ranges
    is not supported.
    This limitation arises because Optuna leverages a dynamic search space, which cannot be
    trivially converted into a static search space as used in DeepCAVE.

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

