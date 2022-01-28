from typing import Optional

import itertools as it

import numpy as np
import pyrfr
import pyrfr.regression as regression
import pyrfr.util
from smac.configspace import ConfigurationSpace

from deepcave.evaluators.epm.forest import Forest


class fANOVAForest(Forest):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        seed: int,
        num_trees: int = 16,
        bootstrapping=True,
        points_per_tree=-1,
        ratio_features: float = 7.0 / 10.0,
        min_samples_split=0,
        min_samples_leaf=0,
        max_depth=64,
        cutoffs=(-np.inf, np.inf),
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = None,
    ):

        super().__init__(configspace, seed, instance_features, pca_components)

        max_features = 0
        if ratio_features <= 1.0:
            max_features = max(1, int(len(self.types) * ratio_features))

        self._set_model_options(
            {
                "num_trees": num_trees,
                "do_bootstrapping": bootstrapping,
                "tree_opts.max_features": max_features,
                "tree_opts.min_samples_to_split": min_samples_split,
                "tree_opts.min_samples_in_leaf": min_samples_leaf,
                "tree_opts.max_depth": max_depth,
            }
        )

        self.cutoffs = cutoffs
        self.points_per_tree = points_per_tree
        self.n_dims = len(configspace.get_hyperparameters())

    def _get_model(self):
        return regression.fanova_forest()

    def _train(self, X, Y):
        """
        Inputs:
            `X`: Must be numerical encoded.
        """

        super()._train(X, Y)
        self.percentiles = np.percentile(Y, range(0, 100))

        # all midpoints and interval sizes treewise for the whole forest
        self.all_midpoints = []
        self.all_sizes = []

        # getting split values
        forest_split_values = self.model.all_split_values()

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
                    # if not, simply append 0 as the value with the number of categories as the size, that way this
                    # parameter will get 0 importance from this tree.
                    else:
                        midpoints.append((0,))
                        sizes.append((self.bounds[i][0],))
                else:
                    # add bounds to split values
                    sv = np.array(
                        [self.bounds[i][0]] + list(split_vals) + [self.bounds[i][1]]
                    )
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

        self._set_cutoffs(self.cutoffs)

        # recompute the trees' total variance
        self.trees_total_variance = self.model.get_trees_total_variances()

    def _set_cutoffs(self, cutoffs=(-np.inf, np.inf), quantile=None):
        """
        Setting the cutoffs to constrain the input space

        To properly do things like 'improvement over default' the
        fANOVA now supports cutoffs on the y values. These will exclude
        parts of the parameters space where the prediction is not within
        the provided cutoffs. This is is specialization of
        "Generalized Functional ANOVA Diagnostics for High Dimensional
        Functions of Dependent Variables" by Hooker.
        """
        if not (quantile is None):
            percentile1 = self.percentiles[quantile[0]]
            percentile2 = self.percentiles[quantile[1]]
            self.model.set_cutoffs(percentile1, percentile2)
        else:
            self.cutoffs = cutoffs
            self.model.set_cutoffs(cutoffs[0], cutoffs[1])

    def compute_marginals(self, dimensions, depth=1):
        """
        Returns the marginal of selected parameters

        Parameters
        ----------
        dimensions: tuple
            Contains the indices of ConfigSpace for the selected parameters (starts with 0)
        """
        dimensions = tuple(dimensions)

        # check if values has been previously computed
        if dimensions in self.V_U_individual:
            return self.V_U_individual, self.V_U_total

        # otherwise make sure all lower order marginals have been
        # computed, if not compute them
        for k in range(1, len(dimensions)):
            if k > depth:
                break

            for sub_dims in it.combinations(dimensions, k):
                if sub_dims not in self.V_U_total:
                    self.compute_marginals(sub_dims)

        # now all lower order terms have been computed
        self.V_U_individual[dimensions] = []
        self.V_U_total[dimensions] = []

        if len(dimensions) > depth + 1:
            return self.V_U_individual, self.V_U_total

        for tree_idx in range(len(self.all_midpoints)):
            # collect all the midpoints and corresponding sizes for that tree
            midpoints = [self.all_midpoints[tree_idx][dim] for dim in dimensions]
            sizes = [self.all_sizes[tree_idx][dim] for dim in dimensions]
            stat = pyrfr.util.weighted_running_stats()

            prod_midpoints = it.product(*midpoints)
            prod_sizes = it.product(*sizes)

            sample = np.full(self.n_dims, np.nan, dtype=np.float)

            # make prediction for all midpoints and weigh them by the corresponding size
            for i, (m, s) in enumerate(zip(prod_midpoints, prod_sizes)):
                sample[list(dimensions)] = list(m)

                ls = self.model.marginal_prediction_stat_of_tree(
                    tree_idx, sample.tolist()
                )
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
                for k in range(1, len(dimensions)):
                    if k > depth:
                        break

                    for sub_dims in it.combinations(dimensions, k):
                        V_U_individual -= self.V_U_individual[sub_dims][tree_idx]
                V_U_individual = np.clip(V_U_individual, 0, np.inf)

            self.V_U_individual[dimensions].append(V_U_individual)
            self.V_U_total[dimensions].append(V_U_total)

        return self.V_U_individual, self.V_U_total
