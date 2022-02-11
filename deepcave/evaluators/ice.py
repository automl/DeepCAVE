from typing import Any, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace

from deepcave.evaluators.epm.random_forest import RandomForest


class ICE:
    def __init__(self, data: Optional[dict[int, Any]] = None):
        self.model = None

        # Make sure to have int keys
        self.data = {}

        if data is not None:
            for k, (X, Y_mean, Y_var) in data.items():
                X = np.array(X)
                Y_mean = np.array(Y_mean)
                Y_var = np.array(Y_var)

                self.data[int(k)] = (X, Y_mean, Y_var)

    def get_data(self) -> dict[int, Any]:
        return self.data

    def fit(
        self, configspace: ConfigurationSpace, X: np.ndarray, Y: np.ndarray, seed=0
    ):
        # Train random forest here
        if self.model is None:
            self.model = RandomForest(
                configspace=configspace,
                seed=seed,
                # num_trees=num_trees,
                bootstrapping=False,
                # points_per_tree=points_per_tree,
                # ratio_features=ratio_features,
                # min_samples_split=min_samples_split,
                # min_samples_leaf=min_samples_leaf,
                # max_depth=max_depth,
                # cutoffs=cutoffs,
            )
            self.model.train(X, Y)

        for hp_name in configspace.get_hyperparameter_names():
            s = configspace.get_idx_by_hyperparameter_name(hp_name)

            shape = X.shape
            X_ice = np.zeros((shape[0], *shape))
            y_ice_mean = np.zeros((shape[0], shape[0]))
            y_ice_var = np.zeros((shape[0], shape[0]))

            # Iterate over data points
            for i, _ in enumerate(X):
                X_copy = X.copy()

                # Intervention
                # Take the value of i-th data point and set it to all others
                # We basically fix the value
                X_copy[:, s] = X_copy[i, s]
                X_ice[i] = X_copy

                # Then we do a prediction with the new data
                # print(self.model.__dict__)
                mean, var = self.model.predict(X_copy)
                var = var.reshape((-1,))
                mean = mean.reshape((-1,))

                y_ice_mean[i] = mean
                y_ice_var[i] = var

            self.data[int(s)] = (X_ice, y_ice_mean, y_ice_var)

    def get_ice_data(
        self, s: int, centered=False, variance_based=False
    ) -> tuple[list[float], list[float]]:
        """
        Args:
            s (int): The id of the requested hyperparameter.
        """
        if s not in self.data:
            return [], []

        (X_ice, y_ice_mean, y_ice_var) = self.data[s]

        if variance_based:
            y_ice = y_ice_var
        else:
            y_ice = y_ice_mean

        all_x = []
        all_y = []

        for i in range(X_ice.shape[0]):
            x = X_ice[:, i, s]
            y = y_ice[:, i]

            # We have to sort x because they might be not
            # in the right order
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

            # Or all zero centered (c-ICE)
            if centered:
                y = y - y[0]

            all_x.append(x)
            all_y.append(y)

        return all_x, all_y

    def get_pdp_data(
        self, s: int, variance_based=False
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Args:
            s (int): The id of the requested hyperparameter.
        """
        if s not in self.data:
            return [], [], []

        (X_ice, y_ice_mean, y_ice_var) = self.data[s]

        if variance_based:
            y_ice = y_ice_var
        else:
            y_ice = y_ice_mean

        n = y_ice.shape[0]

        # Take all x_s instance value
        x = X_ice[:, 0, s]

        y = []
        y_err = []

        # Simply take all values and mean them
        for i in range(n):
            m = np.mean(y_ice[i, :])
            y.append(m)

            std = np.std(y_ice[i, :])
            y_err.append(std)

        y = np.array(y)
        y_err = np.array(y_err)

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        y_err = y_err[idx]

        return x, y, y_err
