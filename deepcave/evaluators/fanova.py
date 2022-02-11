from typing import Optional, Union

import itertools as it
from collections import OrderedDict

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace


class fANOVA:
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Y,
        configspace: ConfigurationSpace,
        seed=0,
        num_trees=16,
        bootstrapping=True,
        points_per_tree=-1,
        ratio_features: float = 7 / 10,
        min_samples_split=0,
        min_samples_leaf=0,
        max_depth=64,
        cutoffs=(-np.inf, np.inf),
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = None,
    ):
        """
        Calculate and provide midpoints and sizes from the forest's
        split values in order to get the marginals

        Parameters
        ------------
        X: matrix with the features, either a np.array or a pd.DataFrame (numerically encoded)

        Y: vector with the response values (numerically encoded)

        configspace : ConfigSpace instantiation

        num_trees: number of trees in the forest to be fit

        seed: seed for the forests randomness

        bootstrapping: whether to bootstrap the data for each tree or not

        points_per_tree: number of points used for each tree
                        (only subsampling if bootstrapping is false)

        ratio_features: number of features to be used at each split, default is 70%

        min_samples_split: minimum number of samples required to attempt to split

        min_samples_leaf: minimum number of samples required in a leaf

        max_depth: maximal depth of each tree in the forest

        cutoffs: tuple of (lower, upper), all values outside this range will be
                 mapped to either the lower or the upper bound. (See:
                 "Generalized Functional ANOVA Diagnostics for High Dimensional
                 Functions of Dependent Variables" by Hooker.)
        """

        self.cs = configspace
        self.cs_params = self.cs.get_hyperparameters()
        self.num_dims = len(self.cs_params)
        self.num_trees = num_trees

        from deepcave.evaluators.epm.fanova_forest import fANOVAForest

        self.forest = fANOVAForest(
            configspace=configspace,
            seed=seed,
            num_trees=num_trees,
            bootstrapping=bootstrapping,
            points_per_tree=points_per_tree,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            cutoffs=cutoffs,
            instance_features=instance_features,
            pca_components=pca_components,
        )

        self.forest.train(X, Y)

    def quantify_importance(
        self, dims, depth=1, sort=True
    ) -> dict[tuple, tuple[float, float, float, float]]:
        """
        Inputs:
            `depth`: How often dims should be combined.

        Returns:
            ordered dict on total importance
            Dict[Tuple[dim_names] -> (
                                mean_fractions_individual,
                                mean_fractions_total,
                                std_fractions_individual,
                                std_fractions_total
                            )]
        """

        if type(dims[0]) == str:
            idx = []
            for i, param in enumerate(dims):
                idx.append(self.cs.get_idx_by_hyperparameter_name(param))
            dimensions = idx
        # make sure that all the V_U values are computed for each tree
        else:
            dimensions = dims

        vu_individual, vu_total = self.forest.compute_marginals(dimensions, depth)

        importance_dict = {}

        for k in range(1, len(dimensions) + 1):
            if k > depth:
                break

            for sub_dims in it.combinations(dimensions, k):
                if type(dims[0]) == str:
                    dim_names = []
                    for j, dim in enumerate(sub_dims):
                        dim_names.append(self.cs.get_hyperparameter_by_idx(dim))
                    dim_names = tuple(dim_names)
                    importance_dict[dim_names] = {}
                else:
                    importance_dict[sub_dims] = {}

                # clean here to catch zero variance in a trees
                non_zero_idx = np.nonzero(
                    [self.forest.trees_total_variance[t] for t in range(self.num_trees)]
                )
                if len(non_zero_idx[0]) == 0:
                    raise RuntimeError("Encountered zero total variance in all trees.")

                fractions_total = np.array(
                    [
                        vu_total[sub_dims][t] / self.forest.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )
                fractions_individual = np.array(
                    [
                        vu_individual[sub_dims][t] / self.forest.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )

                if type(dims[0]) == str:
                    sub_dims = dim_names

                importance_dict[sub_dims] = (
                    np.mean(fractions_individual),
                    np.mean(fractions_total),
                    np.std(fractions_individual),
                    np.std(fractions_total),
                )

        if sort:
            sorted_importance_dict = {
                k: v
                for k, v in sorted(importance_dict.items(), key=lambda item: item[1][1])
            }

            return sorted_importance_dict

        return importance_dict

    def marginal_mean_variance_for_values(self, dimlist, values_to_predict):
        """
        Returns the marginal of selected parameters for specific values

        Parameters
        ----------
        dimlist: list
                Contains the indices of ConfigSpace for the selected parameters
                (starts with 0)
        values_to_predict: list
                Contains the values to be predicted

        Returns
        -------
        tuple
            marginal mean prediction and corresponding variance estimate
        """
        sample = np.full(self.n_dims, np.nan, dtype=np.float)
        for i in range(len(dimlist)):
            sample[dimlist[i]] = values_to_predict[i]

        return self.forest.forest.marginal_mean_variance_prediction(sample)

    def get_most_important_pairwise_marginals(self, params=None, n=10):
        """
        Returns the n most important pairwise marginals from the whole ConfigSpace

        Parameters
        ----------
        params: list of strings or ints
            If specified, limit analysis to those parameters. If ints, interpreting as indices from ConfigurationSpace
        n: int
             The number of most relevant pairwise marginals that will be returned

        Returns
        -------
        list:
             Contains the n most important pairwise marginals
        """
        self.tot_imp_dict = OrderedDict()
        pairwise_marginals = []
        if params is None:
            dimensions = range(self.n_dims)
        else:
            if type(params[0]) == str:
                idx = []
                for i, param in enumerate(params):
                    idx.append(self.cs.get_idx_by_hyperparameter_name(param))
                dimensions = idx

            else:
                dimensions = params
        # pairs = it.combinations(dimensions,2)
        pairs = [x for x in it.combinations(dimensions, 2)]
        if params:
            n = len(list(pairs))
        for combi in pairs:
            pairwise_marginal_performance = self.quantify_importance(combi)
            tot_imp = pairwise_marginal_performance[combi]["individual importance"]
            combi_names = [self.cs_params[combi[0]].name, self.cs_params[combi[1]].name]
            pairwise_marginals.append((tot_imp, combi_names[0], combi_names[1]))

        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)

        for marginal, p1, p2 in pairwise_marginal_performance[:n]:
            self.tot_imp_dict[(p1, p2)] = marginal

        return self.tot_imp_dict

    def get_triple_marginals(self, params=None):
        """
        Returns the n most important pairwise marginals from the whole ConfigSpace

        Parameters
        ----------
        params: list
             The parameters

        Returns
        -------
        list:
             Contains most important triple marginals
        """
        self.tot_imp_dict = OrderedDict()
        triple_marginals = []
        if len(params) < 3:
            raise RuntimeError(
                "Number of parameters have to be greater than %i. At least 3 parameters needed"
                % len(params)
            )
        if type(params[0]) == str:
            idx = []
            for i, param in enumerate(params):
                idx.append(self.cs.get_idx_by_hyperparameter_name(param))
            dimensions = idx

        else:
            dimensions = params

        triplets = [x for x in it.combinations(dimensions, 3)]
        for combi in triplets:
            triple_marginal_performance = self.quantify_importance(combi)
            tot_imp = triple_marginal_performance[combi]["individual importance"]
            combi_names = [
                self.cs_params[combi[0]].name,
                self.cs_params[combi[1]].name,
                self.cs_params[combi[2]].name,
            ]
            triple_marginals.append(
                (tot_imp, combi_names[0], combi_names[1], combi_names[2])
            )

        triple_marginal_performance = sorted(triple_marginals, reverse=True)
        if params:
            triple_marginal_performance = triple_marginal_performance[
                : len(list(triplets))
            ]

        for marginal, p1, p2, p3 in triple_marginal_performance:
            self.tot_imp_dict[(p1, p2, p3)] = marginal

        return self.tot_imp_dict


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "../../")

    import ConfigSpace
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    import numpy as np
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        Constant,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    cs = CS.ConfigurationSpace(seed=1234)

    alpha = CSH.UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
    beta = CSH.UniformFloatHyperparameter(name="beta", lower=0, upper=1)
    gamma = CSH.UniformFloatHyperparameter(name="gamma", lower=0, upper=1)
    gamma1 = CSH.UniformFloatHyperparameter(name="gamma1", lower=0, upper=1)
    gamma2 = CSH.UniformFloatHyperparameter(name="gamma2", lower=0, upper=1)
    gamma3 = CSH.UniformFloatHyperparameter(name="gamma3", lower=0, upper=1)

    # Constants do not work
    # gamma = CSH.Constant(name='gamma', value=1)

    cs.add_hyperparameters([alpha, beta, gamma])

    X = []
    Y = []

    for config in cs.sample_configuration(100):
        cost = np.random.randn()
        encoded = config.get_array()

        X.append(encoded)
        Y.append(cost)

    X = np.array(X)
    Y = np.array(Y)

    conditional = {}
    impute_values = {}

    for idx, hp in enumerate(cs.get_hyperparameters()):
        if idx not in conditional:
            parents = cs.get_parents_of(hp.name)
            if len(parents) == 0:
                conditional[idx] = False
            else:
                conditional[idx] = True
                if isinstance(hp, CategoricalHyperparameter):
                    impute_values[idx] = len(hp.choices)
                elif isinstance(
                    hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)
                ):
                    impute_values[idx] = -1
                elif isinstance(hp, Constant):
                    impute_values[idx] = 1
                else:
                    raise ValueError

        if conditional[idx] is True:
            nonfinite_mask = ~np.isfinite(X[:, idx])
            X[nonfinite_mask, idx] = impute_values[idx]

    # f = fANOVA(X, Y, cs)
    # imp = f.quantify_importance(cs.get_hyperparameter_names()[:3], depth=1)
    # print(imp)

    f = fANOVA(X, Y, cs)
    imp = f.quantify_importance(cs.get_hyperparameter_names(), depth=1, sorted=False)
    print(imp)
