from telnetlib import X3PAD
from typing import Tuple, Union, Optional, List
from itsdangerous import NoneAlgorithm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import MDS
from deepcave.constants import BORDER_CONFIG_ID, RANDOM_CONFIG_ID
from deepcave.runs import AbstractRun, Status
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter

from deepcave.runs.objective import Objective
from deepcave.utils.configspace import get_border_configs


class Footprint:
    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace
        self._model: Optional[RandomForestRegressor] = None
        self._X: Optional[np.ndarray] = None
        self._config_ids: Optional[List[int]] = None
        self._incumbent_id: Optional[int] = None

    def calculate(
        self,
        objective: Objective,
        budget: Union[int, float],
        include_supports: bool = True,
        include_borders: bool = True,
    ) -> None:
        """
        Calculates the distances and trains the model.

        Parameters
        ----------
        objective : Objective
            Objective and colour to show.
        budget : Union[int, float]
            All configurations with this budget are considered.
        include_supports : bool
            Whether random points should be taken into account for MDS.
        include_borders : bool
            Whether border configurations should be taken into account for MDS.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x, y, z for contour plots.
        """
        # Get encoded configs
        data = self.run.get_encoded_data(
            objective, budget, statuses=Status.SUCCESS, specific=True, include_config_ids=True
        )
        hp_names = self.run.configspace.get_hyperparameter_names()

        # Make numpy arrays
        X = data[hp_names].to_numpy()
        Y = data[objective["name"]].to_numpy()
        config_ids = data["config_id"].values.tolist()

        best_y = np.inf
        if objective["optimize"] == "upper":
            best_y = -best_y

        for y, config_id in zip(Y, config_ids):
            if (objective["optimize"] == "lower" and y < best_y) or (
                objective["optimize"] == "upper" and y > best_y
            ):
                best_y = y
                self._incumbent_id = config_id

        # Reshape Y to 2D
        Y = Y.reshape(-1, 1)

        # Create new arrays for sampled, border and random configs
        X_all = X.copy()

        border_configs = get_border_configs(self.cs)
        support_configs = self.cs.sample_configuration(size=len(hp_names) * 10)

        # Add border and support configs to `X_all`
        for configs, config_id in zip(
            [border_configs, support_configs], [BORDER_CONFIG_ID, RANDOM_CONFIG_ID]
        ):
            X_ = self.run.encode_configs(configs)
            X_all = np.concatenate((X_all, X_), axis=0)
            config_ids += [config_id] * len(X_)

        # Get distance between configs
        distances = self._get_distances(X_all)

        # Calculate MDS now to get 2D coordinates
        X_scaled = self._get_mds(distances)

        # But here's the catch: Get rid of border and random configs because
        # we don't have the y values for them.
        self._train(X_scaled[: len(X)], Y.ravel())

        # And we set those points so we can reach them later again
        self._X = X_scaled
        self._config_ids = config_ids

    def get_surface(self, details: float = 0.5) -> Tuple[List, List, List]:
        """
        Get surface of the MDS plot.

        Parameters
        ----------
        details : float, optional
            Steps to create the meshgrid. By default 0.5.

        Returns
        -------
        Tuple[List, List, List]
            x (1D), y (1D) and z (2D) arrays for heatmap.

        Raises
        ------
        RuntimeError
            If `calculate` was not called before.
        """
        X = self._X
        if X is None:
            raise RuntimeError("You need to call `calculate` first.")

        # Create meshgrid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        x = np.arange(x_min, x_max, details)
        y = np.arange(y_min, y_max, details)
        x_mesh, y_mesh = np.meshgrid(x, y)
        conc = np.c_[x_mesh.ravel(), y_mesh.ravel()]

        z = self._model.predict(conc)
        z = z.reshape(x_mesh.shape)

        return x.tolist(), y.tolist(), z.tolist()

    def get_points(self, category="configs") -> Tuple[List[float], List[float], List[int]]:
        """
        Returns the points of the MDS plot.

        Parameters
        ----------
        category : str, optional
            Points of a specific category. Chose between `configs`, `borders`, `supports`
            or `incumbents`. By default `configs`.

        Returns
        -------
        Tuple[List[float], List[float], List[int]]
            X, Y and config_ids as lists.

        Raises
        ------
        RuntimeError
            If category is not supported.
        """
        if category not in ["configs", "borders", "supports", "incumbents"]:
            raise RuntimeError("Unknown category.")

        X = []
        Y = []
        config_ids = []
        for x, config_id in zip(self._X, self._config_ids):
            if (
                (category == "configs" and config_id >= 0)
                or (category == "borders" and config_id == BORDER_CONFIG_ID)
                or (category == "incumbents" and config_id == self._incumbent_id)
                or (category == "supports" and config_id == RANDOM_CONFIG_ID)
            ):
                x = x.tolist()  # type: ignore
                X += [x[0]]
                Y += [x[1]]
                config_ids += [config_id]

        return X, Y, config_ids

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

        is_categorical = np.array(is_categorical)  # type: ignore
        depth = np.array(depth)  # type: ignore

        for i in range(n_configs):
            for j in range(i + 1, n_configs):
                d = np.abs(X[i, :] - X[j, :])
                d[np.isnan(d)] = 1
                d[np.logical_and(is_categorical, d != 0)] = 1
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

    def _get_mds(self, distances: np.ndarray) -> np.ndarray:
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
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
        X_scaled = mds.fit_transform(distances)

        return X_scaled

    def _train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the random forest on the performance.

        Parameters
        ----------
        X : np.ndarray
            Numpy array with MDS coordinates in 2D.
        Y : np.ndarray
            Numpy array with costs.
        """
        self._model = RandomForestRegressor(random_state=0)
        self._model.fit(X, Y)
