from abc import abstractmethod

import functools

import numpy as np
import pyrfr.regression as regression
from smac.configspace import ConfigurationSpace
from smac.epm.base_rf import BaseModel
from smac.epm.util_funcs import get_types


class Forest(BaseModel):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        seed=0,
        instance_features=None,
        pca_components=None,
    ):
        # Set types and bounds automatically
        types, bounds = get_types(configspace, instance_features)

        self.types = types
        self.bounds = bounds
        self.points_per_tree = None

        super().__init__(
            configspace=configspace,
            types=types,
            bounds=bounds,
            seed=seed,
            instance_features=instance_features,
            pca_components=pca_components,
        )

        self.rng = regression.default_random_engine(seed)

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError()

    def _set_model_options(self, d):
        self.model_options = regression.forest_opts()

        def rgetattr(obj, attr, *args):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)

            return functools.reduce(_getattr, [obj] + attr.split("."))

        def rsetattr(obj, attr, val):
            pre, _, post = attr.rpartition(".")
            return setattr(rgetattr(obj, pre) if pre else obj, post, val)

        for k, v in d.items():
            rsetattr(self.model_options, k, v)

    def _train(self, X: np.ndarray, y: np.ndarray) -> "Forest":
        """Trains the random forest on X and y.
        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        y : np.ndarray [n_samples, ]
            The corresponding target values.
        Returns
        -------
        self
        """
        # X = self._impute_inactive(X)
        data = self._init_data_container(X, y.flatten())

        self.model_options.num_data_points_per_tree = X.shape[0]
        if self.points_per_tree is not None and self.points_per_tree > 0:
            self.model_options.num_data_points_per_tree = self.points_per_tree

        self.model = self._get_model()
        self.model.options = self.model_options
        self.model.fit(data, rng=self.rng)

        return self

    def _init_data_container(
        self, X: np.ndarray, y: np.ndarray
    ) -> regression.default_data_container:
        """Fills a pyrfr default data container, s.t. the forest knows
        categoricals and bounds for continous data
        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points
        y : np.ndarray [n_samples, ]
            Corresponding target values
        Returns
        -------
        data : regression.default_data_container
            The filled data container that pyrfr can interpret
        """

        # retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self.bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)

        return data
