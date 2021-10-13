
import glob
import os
import numpy as np
import time
from deep_cave.runs.run import Status, Run
from deep_cave.utils.files import make_dirs


class Recorder:
    def __init__(self,
                 configspace,
                 objectives="cost",
                 objective_weights=None,
                 meta={},
                 save_path="logs",
                 trial_name=None):
        """
        All objectives follow the scheme the lower the better.
        If file

        Parameters:
            save_path (str): Blub.
            configspace (ConfigSpace):
            objectives (str or list):
            trial_name: Name of the trial. If not given, trial_x will be used.
        """

        self._set_path(save_path, trial_name)

        # Set variables
        self.last_trial_id = None
        self.start_time = time.time()
        self.start_times = {}
        self.models = {}
        self.origins = {}
        self.additionals = {}

        # Define trials container
        self.run = Run(
            configspace=configspace,
            objectives=objectives,
            objective_weights=objective_weights,
            meta=meta
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def _set_path(self, path, name):
        """
        Identifies the latest run and sets the path with increased id.
        """

        # Make sure the word is interpreted as folder
        if path[-1] != "/":
            make_dirs(path + "/")
        else:
            make_dirs(path)
            # Remove last slash
            path = path[:-1]

        if name is None:
            new_idx = 0
            for file in glob.glob(f"{path}/run_*"):
                idx = file.split("_")[-1]
                if idx.isnumeric():
                    idx = int(idx)
                    if idx > new_idx:
                        new_idx = idx

            # And increase the id
            new_idx += 1

            self.path = os.path.join(path, f"run_{new_idx}")

        else:
            self.path = os.path.join(path, name)

            # Check if trial name is already given
            if os.path.exists(self.path):
                raise RuntimeError(
                    f"The path `{self.path}` already exists.")

    def start(self,
              config,
              budget=None,
              model=None,
              origin=None,
              additional={}):

        id = (config, budget)

        # Start timer
        self.start_times[id] = time.time() - self.start_time
        self.models[id] = model
        self.origins[id] = origin
        self.additionals[id] = additional

        self.last_trial_id = id

    def end(self,
            costs=np.inf,
            status=Status.SUCCESS,
            config=None,
            budget=None,
            additional={}):
        """
        In case of multi-processing, config+budget should be passed as otherwise
        it can't be matched correctly.
        """

        if config is not None:
            id = (config, budget)
        else:
            id = self.last_trial_id
            config, budget = id[0], id[1]

        model = self.models[id]
        start_additional = self.additionals[id].copy()
        start_additional.update(additional)

        if costs == np.inf:
            status = Status.CRASHED

        start_time = self.start_times[id]
        end_time = time.time() - self.start_time

        # Add to trial history
        self.run.add(
            costs=costs,
            config=config,
            budget=budget,
            start_time=start_time,
            end_time=end_time,
            status=status,
            model=model,
            additional=start_additional
        )

        # Clean the dicts
        del self.start_times[id]
        del self.models[id]
        del self.origins[id]
        del self.additionals[id]

        # And save the results
        self.run.save(self.path)
