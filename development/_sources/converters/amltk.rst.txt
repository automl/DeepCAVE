AMLTK
-----

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