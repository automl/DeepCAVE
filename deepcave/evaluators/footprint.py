from typing import Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import MDS
from deepcave.runs import AbstractRun, Status
from ConfigSpace import Hyperparameter, CategoricalHyperparameter

from deepcave.runs.objective import Objective


class Footprint:
    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace

    def calculate(
        self, objective: Objective, budget: Union[int, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the MDS and returns the contour data.

        Parameters
        ----------
        objective : Objective
            Objective and colour to show.
        budget : Union[int, float]
            All configurations with this budget are considered.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x, y, z for contour plots.
        """

        # Get encoded configs
        X, Y = self.run.get_encoded_configs(
            objectives=[objective],
            budget=budget,
            statuses=[Status.SUCCESS],
            specific=True,
            pandas=False,
        )

        # Get distance between configs
        distances = self._get_distances(X)

        # Calculate MDS now
        X_scaled = self._perform_mds(distances)

        return self._get_surface(X_scaled, Y)

    def _get_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the distance between all pairs of configurations.

        Parameters
        ----------
        X : np.ndarray
            Numpy array with encoded configurations.

        Returns
        -------
        np.ndarray
            np.array with distances between configurations i,j in dists[i,j] or dists[j,i].
        """

        n_configs = X.shape[0]
        distances = np.zeros((n_configs, n_configs))

        is_categorical = []
        depth = []
        for hp in self.cs.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                is_categorical.append(True)
            else:
                is_categorical.append(False)
            depth.append(self._get_depth(hp))

        is_categorical = np.array(is_categorical)
        depth = np.array(depth)

        for i in range(n_configs):
            for j in range(i + 1, n_configs):
                d = np.abs(X[i, :] - X[j, :])
                d[np.isnan(d)] = 1
                d[np.logical_and(is_categorical, distances != 0)] = 1
                d = np.sum(d / depth)
                distances[i, j] = d
                distances[j, i] = d

        return distances

    def _get_depth(self, hp: Hyperparameter) -> int:
        """
        Get depth (generations above) in configuration space of a given hyperparameter.

        Parameters
        ----------
        param: str
            name of parameter to inspect

        Returns
        int
            Depth of the hyperparameter.
        """

        parents = self.cs.get_parents_of(hp)
        if not parents:
            return 1

        new_parents = parents
        d = 1
        while new_parents:
            d += 1
            old_parents = new_parents
            new_parents = []
            for p in old_parents:
                pp = self.cs.get_parents_of(p)
                if pp:
                    new_parents.extend(pp)
                else:
                    return d

        return d

    def _perform_mds(self, distances: np.ndarray) -> np.ndarray:
        """
        Perform MDS on the distances.

        Parameters
        ----------
        distances : np.ndarray
            Numpy array with distances between configurations.

        Returns
        -------
        np.ndarray
            Numpy array with MDS coordinates in 2D.
        """

        # Perform MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
        X_scaled = mds.fit_transform(distances)

        return X_scaled

    def _get_surface(
        self, X_scaled: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get surface of the MDS plot.

        Parameters
        ----------
        X_scaled : np.ndarray
            Numpy array with MDS coordinates in 2D.
        Y : np.ndarray
            Numpy array with costs.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x, y, z for contour plots.
        """

        # Train a random forest
        model = RandomForestRegressor()
        model.fit(X_scaled, Y)

        # Create meshgrid
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        x, y = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

        # Predict the values
        z = model.predict(np.c_[x.ravel(), y.ravel()])

        return x, y, z
