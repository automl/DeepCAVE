from typing import List, Optional, Tuple

import itertools as it

import numpy as np
import pyrfr
import pyrfr.regression as regression
import pyrfr.util
from ConfigSpace import ConfigurationSpace

from deepcave.evaluators.epm.random_forest import RandomForest


class FanovaForest(RandomForest):
    """
    A fanova forest wrapper for pyrfr.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        n_trees: int = 10,
        ratio_features: float = 1.0,
        min_samples_split: int = 0,
        min_samples_leaf: int = 0,
        max_depth: int = 64,
        max_nodes: int = 2**20,
        eps_purity: float = 1e-8,
        bootstrapping: bool = True,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = 2,
        cutoffs: Tuple[float, float] = (-np.inf, np.inf),
        seed: Optional[int] = None,
    ):
        super().__init__(
            configspace=configspace,
            n_trees=n_trees,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_nodes=max_nodes,
            eps_purity=eps_purity,
            bootstrapping=bootstrapping,
            instance_features=instance_features,
            pca_components=pca_components,
            log_y=False,
            seed=seed,
        )

        self.cutoffs = cutoffs

    def _get_model(self) -> regression.base_tree:
        """
        Returns the internal model.

        Returns
        -------
        model : regression.base_tree
            Model which is used internally.
        """
        return regression.fanova_forest()

    def _train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the random forest on X and Y.

        Parameters
        ----------
        X : np.ndarray
            Input data points.
        Y : np.ndarray
            Target values.
        """
        super()._train(X, Y)
        self.percentiles = np.percentile(Y, range(0, 100))

        # all midpoints and interval sizes treewise for the whole forest
        self.all_midpoints = []
        self.all_sizes = []

        # getting split values
        forest_split_values = self._model.all_split_values()

        # compute midpoints and interval sizes for variables in each tree
        for tree_split_values in forest_split_values:
            sizes = []
            midpoints = []
            for i, split_vals in enumerate(tree_split_values):
                if np.isnan(self.bounds[i][1]):  # categorical parameter
                    # check if the tree actually splits on this parameter
                    if len(split_vals) > 0:
                        midpoints.append(split_vals)
                        sizes.append(np.ones(len(split_vals)))
                    # if not, simply append 0 as the value with the number of categories as the
                    # size, that way this parameter will get 0 importance from this tree.
                    else:
                        midpoints.append((0,))
                        sizes.append((self.bounds[i][0],))
                else:
                    # add bounds to split values
                    sv = np.array([self.bounds[i][0]] + list(split_vals) + [self.bounds[i][1]])
                    # compute midpoints and sizes
                    midpoints.append((1 / 2) * (sv[1:] + sv[:-1]))
                    sizes.append(sv[1:] - sv[:-1])

            self.all_midpoints.append(midpoints)
            self.all_sizes.append(sizes)

        # capital V in the paper
        self.trees_total_variances = []

        # dict of lists where the keys are tuples of the dimensions
        # and the value list contains \hat{f}_U for the individual trees
        # reset all the variance fractions computed
        self.trees_variance_fractions = {}
        self.V_U_total = {}
        self.V_U_individual = {}

        # Set cut-off
        self._model.set_cutoffs(self.cutoffs[0], self.cutoffs[1])

        # recompute the trees' total variance
        self.trees_total_variance = self._model.get_trees_total_variances()

    def compute_marginals(self, hp_ids: List[int], depth=1):
        """
        Returns the marginal of selected parameters.

        Parameters
        ----------
        hp_ids: List[int]
            Contains the indices of the configspace for the selected parameters (starts with 0).
        """
        hp_ids = tuple(hp_ids)

        # check if values has been previously computed
        if hp_ids in self.V_U_individual:
            return self.V_U_individual, self.V_U_total

        # otherwise make sure all lower order marginals have been
        # computed, if not compute them
        for k in range(1, len(hp_ids)):
            if k > depth:
                break

            for sub_hp_ids in it.combinations(hp_ids, k):
                if sub_hp_ids not in self.V_U_total:
                    self.compute_marginals(sub_hp_ids)

        # now all lower order terms have been computed
        self.V_U_individual[hp_ids] = []
        self.V_U_total[hp_ids] = []

        if len(hp_ids) > depth + 1:
            return self.V_U_individual, self.V_U_total

        for tree_idx in range(len(self.all_midpoints)):
            # collect all the midpoints and corresponding sizes for that tree
            midpoints = [self.all_midpoints[tree_idx][hp_id] for hp_id in hp_ids]
            sizes = [self.all_sizes[tree_idx][hp_id] for hp_id in hp_ids]
            stat = pyrfr.util.weighted_running_stats()

            prod_midpoints = it.product(*midpoints)
            prod_sizes = it.product(*sizes)

            sample = np.full(self.n_params, np.nan, dtype=float)

            # make prediction for all midpoints and weigh them by the corresponding size
            for i, (m, s) in enumerate(zip(prod_midpoints, prod_sizes)):
                sample[list(hp_ids)] = list(m)

                ls = self._model.marginal_prediction_stat_of_tree(tree_idx, sample.tolist())
                # self.logger.debug("%s, %s", (sample, ls.mean()))
                if not np.isnan(ls.mean()):
                    stat.push(ls.mean(), np.prod(np.array(s)) * ls.sum_of_weights())

            # line 10 in algorithm 2
            # note that V_U^2 can be computed by var(\hat a)^2 - \sum_{subU} var(f_subU)^2
            # which is why, \hat{f} is never computed in the code, but
            # appears in the pseudocode
            V_U_total = np.nan
            V_U_individual = np.nan

            if stat.sum_of_weights() > 0:
                V_U_total = stat.variance_population()
                V_U_individual = stat.variance_population()
                for k in range(1, len(hp_ids)):
                    if k > depth:
                        break

                    for sub_dims in it.combinations(hp_ids, k):
                        V_U_individual -= self.V_U_individual[sub_dims][tree_idx]
                V_U_individual = np.clip(V_U_individual, 0, np.inf)

            self.V_U_individual[hp_ids].append(V_U_individual)
            self.V_U_total[hp_ids].append(V_U_total)

        return self.V_U_individual, self.V_U_total
