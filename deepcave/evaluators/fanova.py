#  noqa: D400
"""
# fANOVA

This module provides a tool for assessing the importance of an algorithms Hyperparameters.

Utilities provide calculation of the data wrt the budget and train the forest on the encoded data.

## Classes
    - fANOVA: Calculate and provide midpoints and sizes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import itertools as it

import numpy as np

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.logs import get_logger


class fANOVA:
    """
    Calculate and provide midpoints and sizes.

    They are generated from the forest's split values in order to get the marginals.

    Properties
    ----------
    run : AbstractRun
        The Abstract Run used for the calculation.
    cs : ConfigurationSpace
        The configuration space of the run.
    hps : List[Hyperparameters]
        The Hyperparameters of the configuration space.
    hp_names : List[str]
        The corresponding names of the Hyperparameters.
    n_trees : int
        The number of trees.
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace
        self.hps = self.cs.get_hyperparameters()
        self.hp_names = self.cs.get_hyperparameter_names()
        self.logger = get_logger(self.__class__.__name__)

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        n_trees: int = 16,
        seed: int = 0,
    ) -> None:
        """
        Get the data with respect to budget and train the forest on the encoded data.

        Note
        ----
        Right now, only `n_trees` is used. It can be further specified if needed.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. By default None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default None. If None, the highest budget is chosen.
        n_trees : int, optional
            How many trees should be used. By default 16.
        seed : int
            Random seed. By default 0.
        """
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        self.n_trees = n_trees

        # Get data
        df = self.run.get_encoded_data(
            objectives, budget, specific=True, include_combined_cost=True
        )
        X = df[self.hp_names].to_numpy()
        # Combined cost name includes the cost of all selected objectives
        Y = df[COMBINED_COST_NAME].to_numpy()

        # Get model and train it
        self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
        self._model.train(X, Y)

    def get_importances(
        self, hp_names: Optional[List[str]] = None, depth: int = 1, sort: bool = True
    ) -> Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]]:
        """
        Return the importance scores from the passed Hyperparameter names.

        Warning
        -------
        Using a depth higher than 1 might take much longer.

        Parameters
        ----------
        hp_names : Optional[List[str]]
            Selected Hyperparameter names to get the importance scores from. If None, all
            Hyperparameters of the configuration space are used.
        depth : int, optional
            How often dimensions should be combined. By default 1.
        sort : bool, optional
            Whether the Hyperparameters should be sorted by importance. By default True.

        Returns
        -------
        Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]]
            Dictionary with Hyperparameter names and the corresponding importance scores.
            The values are tuples of the form (mean individual, var individual, mean total,
            var total). Note that individual and total are the same if depth is 1.

        Raises
        ------
        RuntimeError
            If there is zero total variance in all trees.
        """
        if hp_names is None:
            hp_names = self.cs.get_hyperparameter_names()

        hp_ids = []
        for hp_name in hp_names:
            hp_ids.append(self.cs.get_idx_by_hyperparameter_name(hp_name))

        # Calculate the marginals
        vu_individual, vu_total = self._model.compute_marginals(hp_ids, depth)

        importances: Dict[Tuple[Any, ...], Tuple[float, float, float, float]] = {}
        for k in range(1, len(hp_ids) + 1):
            if k > depth:
                break

            for sub_hp_ids in it.combinations(hp_ids, k):
                sub_hp_ids = tuple(sub_hp_ids)

                # clean here to catch zero variance in a trees
                non_zero_idx = np.nonzero(
                    [self._model.trees_total_variance[t] for t in range(self.n_trees)]
                )

                if len(non_zero_idx[0]) == 0:
                    self.logger.warning("Encountered zero total variance in all trees.")
                    importances[sub_hp_ids] = (
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    )
                    continue

                fractions_total = np.array(
                    [
                        vu_total[sub_hp_ids][t] / self._model.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )
                fractions_individual = np.array(
                    [
                        vu_individual[sub_hp_ids][t] / self._model.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )

                importances[sub_hp_ids] = (
                    np.mean(fractions_individual),
                    np.var(fractions_individual),
                    np.mean(fractions_total),
                    np.var(fractions_total),
                )

        # Sort by total mean fraction
        if sort:
            importances = {
                k: v for k, v in sorted(importances.items(), key=lambda item: item[1][2])
            }

        # The ids get replaced with hyperparameter names again
        all_hp_names = self.cs.get_hyperparameter_names()
        importances_: Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]] = {}
        for hp_ids_importances, values in importances.items():
            hp_names = [all_hp_names[hp_id] for hp_id in hp_ids_importances]
            hp_names_key: Union[Tuple[str, ...], str]
            if len(hp_names) == 1:
                hp_names_key = hp_names[0]
            else:
                hp_names_key = tuple(hp_names)
            importances_[hp_names_key] = values

        return importances_

    '''
    def marginal_mean_variance_for_values(self, dimlist, values_to_predict):
        """
        Return the marginal of selected parameters for specific values

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

        return self._model.forest.marginal_mean_variance_prediction(sample)

    def get_most_important_pairwise_marginals(self, params=None, n=10):
        """
        Return the n most important pairwise marginals from the whole ConfigSpace.

        Parameters
        ----------
        params: list of strings or ints
            If specified, limit analysis to those parameters. If ints, interpreting as indices from
            ConfigurationSpace
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
            combi_names = [self.hps[combi[0]].name, self.hps[combi[1]].name]
            pairwise_marginals.append((tot_imp, combi_names[0], combi_names[1]))

        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)

        for marginal, p1, p2 in pairwise_marginal_performance[:n]:
            self.tot_imp_dict[(p1, p2)] = marginal

        return self.tot_imp_dict

    def get_triple_marginals(self, params=None):
        """
        Return the n most important pairwise marginals from the whole ConfigSpace

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
                self.hps[combi[0]].name,
                self.hps[combi[1]].name,
                self.hps[combi[2]].name,
            ]
            triple_marginals.append((tot_imp, combi_names[0], combi_names[1], combi_names[2]))

        triple_marginal_performance = sorted(triple_marginals, reverse=True)
        if params:
            triple_marginal_performance = triple_marginal_performance[: len(list(triplets))]

        for marginal, p1, p2, p3 in triple_marginal_performance:
            self.tot_imp_dict[(p1, p2, p3)] = marginal

        return self.tot_imp_dict
    '''
