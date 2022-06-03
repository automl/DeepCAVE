from typing import List, Optional, Tuple, Union

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import MDS
from tqdm import tqdm

from deepcave.constants import BORDER_CONFIG_ID, RANDOM_CONFIG_ID
from deepcave.runs import AbstractRun, Status
from deepcave.runs.objective import Objective
from deepcave.utils.configspace import sample_border_config, sample_random_config
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class Footprint:
    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace

        # Important parameters
        is_categorical, depth = [], []
        for hp in self.cs.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                is_categorical.append(True)
            else:
                is_categorical.append(False)
            depth.append(self._get_depth(hp))

        self._is_categorical = np.array(is_categorical)  # type: ignore
        self._depth = np.array(depth)  # type: ignore

        # Global variables
        self._distances = None
        self._trained = False
        self._reset()

    def _reset(self) -> None:
        self._objective_model: Optional[RandomForestRegressor] = None
        self._area_model: Optional[RandomForestRegressor] = None
        self._config_ids: Optional[List[int]] = None
        self._incumbent_id: Optional[int] = None

        # Those are used to fit the MDS (consists of random and border configs).
        self._X: Optional[np.ndarray] = None
        # Those are the fitted configs with shape (x, 2).
        self._MDS_X: Optional[np.ndarray] = None

    def calculate(
        self,
        objective: Objective,
        budget: Union[int, float],
        support_discretization: Optional[int] = 10,
        rejection_rate: float = 0.01,
        retries: int = 3,
        exclude_configs: bool = False,
    ) -> None:
        """
        Calculates the distances and trains the model.

        Parameters
        ----------
        objective : Objective
            Objective and colour to show.
        budget : Union[int, float]
            All configurations with this budget are considered.
        support_discretization : Optional[int], optional
            Discretization steps for integer and float hyperparameter values.
        rejection_rate : float, optional
            Rejection rate whether a configuration should be rejected or not. Internally,
            the max distance is calculated and if a configuration has a distance smaller than
            max distance * rejection_rate, the configuration is rejected.
        retries : int, optional
            How many times to retry adding a new configuration.
        exclude_configs : bool, optional
            Whether the configurations from the run should be excluded in the MDS scaling.
            This is particullary interseting if only the search space should be plotted.
        """
        # Reset everything
        self._reset()
        self.cs.seed(0)

        # Get config rejection threshold
        # If the distance between two configs is smaller than the threshold, the config
        # is rejected
        rejection_threshold = self._get_max_distance() * rejection_rate

        # Get encoded configs
        data = self.run.get_encoded_data(
            objective, budget, statuses=Status.SUCCESS, specific=False, include_config_ids=True
        )
        hp_names = self.run.configspace.get_hyperparameter_names()

        # Make numpy arrays
        X = data[hp_names].to_numpy()
        Y = data[objective.name].to_numpy()
        config_ids = data["config_id"].values.tolist()  # type: ignore

        # Get the incumbent
        incumbent_config, _ = self.run.get_incumbent(objective, budget)
        self._incumbent_id = self.run.get_config_id(incumbent_config)

        # Reshape Y to 2D
        Y = Y.reshape(-1, 1)

        # Init distances
        self._init_distances(X, config_ids, exclude_configs=exclude_configs)

        border_generator = sample_border_config(self.cs)
        random_generator = sample_random_config(self.cs, d=support_discretization)

        # Now we add the border and random configs
        count_border = 0
        count_random = 0
        tries = 0
        logger.info("Starting to calculate distances and add border and random configurations...")
        while True:
            _configs = []
            _config_ids = []

            try:
                _configs += [next(border_generator)]
                _config_ids += [BORDER_CONFIG_ID]
            except StopIteration:
                pass

            try:
                _configs += [next(random_generator)]
                _config_ids += [RANDOM_CONFIG_ID]
            except StopIteration:
                pass

            counter = 0
            for config, config_id in zip(_configs, _config_ids):
                if config is None:
                    continue

                # Encode config
                config = np.array(self.run.encode_config(config))
                rejected = self._update_distances(config, config_id, rejection_threshold)
                if not rejected:
                    # Count
                    if config_id == BORDER_CONFIG_ID:
                        count_border += 1

                    if config_id == RANDOM_CONFIG_ID:
                        count_random += 1

                    counter += 1

            # Abort criteria
            # If we don't get new configs
            if counter == 0:
                tries += 1
            else:
                tries = 0

            if tries >= retries:
                break

            # Or if we reach more than 4000 (otherwise it takes too long)
            assert self._distances is not None

            if self._distances.shape[0] % 100 == 0:
                logger.info(f"Found {self._distances.shape[0]} configurations...")

            if self._distances.shape[0] > 4000:
                break

        assert self._distances is not None
        logger.info(f"Added {count_border} border configs and {count_random} random configs.")
        logger.info(f"Total configurations: {self._distances.shape[0]}.")
        logger.info("Getting MDS data...")

        # Calculate MDS now to get 2D coordinates and set those points to reach them later.
        MDS_X = self._get_mds()
        self._MDS_X = MDS_X

        # But here's the catch: Get rid of border and random configs because
        # we don't have the y values for them.
        # However, it makes no sense to train the RF if we exclude the configs
        # which were evaluated.
        if not exclude_configs:
            self._train_on_objective(MDS_X[: len(X)], Y.ravel())
            self._trained = True
        else:
            self._trained = False

        # Train on areas can be done anytime.
        self._train_on_areas()

    def get_surface(
        self, details: float = 0.5, performance: bool = True
    ) -> Tuple[List, List, List]:
        """
        Get surface of the MDS plot.

        Parameters
        ----------
        details : float, optional
            Steps to create the meshgrid. By default 0.5.
        performance : bool, optional
            Whether to get the surface from the performance or the valid areas.

        Returns
        -------
        Tuple[List, List, List]
            x (1D), y (1D) and z (2D) arrays for heatmap.

        Raises
        ------
        RuntimeError
            If `calculate` was not called before.
        """
        X = self._MDS_X
        if X is None:
            raise RuntimeError("You need to call `calculate` first.")

        if performance and not self._trained:
            raise RuntimeError(
                "You can not get the surface" "if you do not include evaluated configs."
            )

        # Create meshgrid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        num = int(20 * details) + 10
        x = np.linspace(x_min, x_max, num)
        y = np.linspace(y_min, y_max, num)
        x_mesh, y_mesh = np.meshgrid(x, y)
        conc = np.c_[x_mesh.ravel(), y_mesh.ravel()]

        if performance:
            model = self._objective_model
        else:
            model = self._area_model

        if model is None:
            raise RuntimeError("You need to call `calculate` first.")

        z = model.predict(conc)
        z = z.reshape(x_mesh.shape)

        return x.tolist(), y.tolist(), z.tolist()

    def get_points(self, category: str = "configs") -> Tuple[List[float], List[float], List[int]]:
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

        if self._MDS_X is None or self._config_ids is None:
            raise RuntimeError("You need to call `calculate` first.")

        X = []
        Y = []
        config_ids = []
        for x, config_id in zip(self._MDS_X, self._config_ids):
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

    def _get_max_distance(self) -> float:
        """
        Calculate the maximum distance between all configs.
        Basically, we just count the number of hps.

        Returns
        -------
        float
            Maximal distance between two configurations.
        """
        # We just count the number of hps
        # Since X is normalized, we can just sum 1
        max_distance = 0
        for hp in self.cs.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter) or isinstance(hp, Constant):
                continue

            max_distance += 1

        return max_distance

    def _get_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates distance between x and y. Both arrays must have the same length.

        Parameters
        ----------
        x : np.ndarray
            Configuration 1.
        y : np.ndarray
            Configuration 2.

        Returns
        -------
        float
            Distance from configuration 1 and configuration 2.
        """
        if self._depth is None or self._is_categorical is None:
            raise RuntimeError("You need to call `calculate` first.")

        d = np.abs(x - y)
        d[np.isnan(d)] = 1
        d[np.logical_and(self._is_categorical, d != 0)] = 1
        d = np.sum(d / self._depth)

        return d

    def _get_distances(self, X: np.ndarray) -> np.ndarray:
        n_configs = X.shape[0]

        # We initiate the distances
        distances = np.zeros((n_configs, n_configs))

        for i in tqdm(range(n_configs)):
            for j in range(i + 1, n_configs):
                d = self._get_distance(X[i, :], X[j, :])
                distances[i, j] = d
                distances[j, i] = d

        return distances

    def _init_distances(
        self, X: np.ndarray, config_ids: List[int], exclude_configs: bool = False
    ) -> None:
        """
        Initializes the distances.

        Parameters
        ----------
        X : np.ndarray
            Encoded data.
        config_ids : List[int]
            Corresponding config_ids.
        exclude_configs : bool, optional
            Whether the passed X should be used or not. By default False.
        """
        if not exclude_configs:
            self._X = X.copy()
            self._config_ids = config_ids
            self._distances = self._get_distances(X)
            logger.info(f"Added {len(config_ids)} configurations.")
        else:
            self._X = None
            self._config_ids = []
            self._distances = np.zeros((0, 0))

    def _update_distances(
        self,
        config: np.ndarray,
        config_id: int,
        rejection_threshold: Optional[float] = 0.0,
    ) -> bool:
        """
        Updates the internal distance if the passed config is not rejectded.

        Parameters
        ----------
        config : np.ndarray
            Config, which is tried to be added.
        config_id : int
            Corresponding config id. This is important for later identification as the config might
            be a border or random configuration.
        rejection_threshold : Optional[float], optional
            Threshold for rejecting the config. By default 0.0.

        Returns
        -------
        rejected : bool
            Whether the config was rejected or not.
        """
        X = self._X
        distances = self._distances

        if X is None:
            X = np.array([[]])
            n_configs = 0
        else:
            n_configs = X.shape[0]

        rejected = False
        new_distances = np.zeros((n_configs + 1, n_configs + 1))

        # In case X is not set
        if n_configs == 0:
            new_distances[0, 0] = 0
        else:
            # Calculate distance to all configs
            new_distances[:n_configs, :n_configs] = distances[:, :]
            for j in range(n_configs):
                d = self._get_distance(X[j, :], config)
                if rejection_threshold is not None:
                    if d < rejection_threshold:
                        rejected = True
                        break

                # Add to new distances
                new_distances[n_configs, j] = d
                new_distances[j, n_configs] = d

        if not rejected:
            # Add to X here
            if X.shape[1] == 0:
                X = np.array([config])
            else:
                X = np.concatenate((X, np.array([config])), axis=0)

            self._X = X
            self._config_ids += [config_id]
            self._distances = new_distances

        return rejected

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

    def _get_mds(self) -> np.ndarray:
        """
        Perform MDS on the internal distances.

        Parameters
        ----------
        distances : np.ndarray
            Numpy array with distances between configurations.

        Returns
        -------
        np.ndarray
            Numpy array with MDS coordinates in 2D.
        """
        if self._distances is None:
            raise RuntimeError("You need to call `calculate` first.")

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
        return mds.fit_transform(self._distances)

    def _train_on_objective(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the random forest on the performance.

        Parameters
        ----------
        X : np.ndarray
            Numpy array with MDS coordinates in 2D.
        Y : np.ndarray
            Numpy array with costs.
        """
        logger.info("Training on objective...")
        self._objective_model = RandomForestRegressor(random_state=0)
        self._objective_model.fit(X, Y)

    def _train_on_areas(self) -> None:
        """
        Trains the random forest on the "valid" areas.
        """
        if self._MDS_X is None:
            raise RuntimeError("You need to call `calculate` first.")

        logger.info("Training on area...")
        MDS_X = self._MDS_X

        # We basically have to create a grid here
        x_min, x_max = MDS_X[:, 0].min() - 1, MDS_X[:, 0].max() + 1
        y_min, y_max = MDS_X[:, 1].min() - 1, MDS_X[:, 1].max() + 1

        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)

        X = []
        Y = []
        for x1, x2 in zip(x, x[1:]):
            for y1, y2 in zip(y, y[1:]):
                # Find center
                center = [(x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1]
                value = 0
                for a, b in MDS_X:
                    # If it's in center add
                    if a >= x1 and a <= x2 and b >= y1 and b <= y2:
                        value = 1
                        break

                X.append(center)
                Y.append(value)

        X = np.array(X)  # type: ignore
        Y = np.array(Y)  # type: ignore

        # Train the model
        self._area_model = RandomForestRegressor(random_state=0)
        self._area_model.fit(X, Y)
