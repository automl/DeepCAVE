Optuna
------

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

Furthermore, to load an Optuna run into DeepCAVE, it is necessary to install Optuna, e.g. via:

   .. code-block:: bash

        pip install deepcave[optuna]

.. warning::
    Loading Optuna runs with conditional search spaces or dynamic hyperparameter value ranges
    is not supported.
    This limitation arises because Optuna leverages a dynamic search space, which cannot be
    trivially converted into a static search space as used in DeepCAVE.
