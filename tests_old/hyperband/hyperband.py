from tqdm import tqdm
import math

import sys
from os.path import dirname, abspath

import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))

from libs.LCBench.api import Benchmark
from tests.hyperband.successive_halving import successive_halving  # noqa: E402
from tests.hyperband.utils import plot_grey_box_optimization  # noqa: E402


def hyperband(problem, min_budget_per_model, max_budget_per_model, eta, random_seed, dataset_name, factor=1):
    """ The hyperband algorithm

    Parameters
    ----------
    problem : instance of Problem
    min_budget_per_model : int
    max_budget_per_model : int
    eta : float
    random_seed : int

    Returns
    -------

    """
    s_max = np.floor(np.log(max_budget_per_model / min_budget_per_model) / np.log(eta)).astype(int)
    budgets = [int(eta ** (-s) * max_budget_per_model) for s in reversed(range(s_max + 1))]
    n_models = [int(np.ceil((s_max + 1) * eta ** s).astype(int)) for s in reversed(range(s_max + 1))]

    configs_dicts = []
    for i, (models, budget) in enumerate(zip(n_models, budgets)):

        configs_dict = successive_halving(problem=problem, n_models=models,
                                          budgets=budgets[i:], eta=eta, random_seed=random_seed,
                                          dataset_name=dataset_name, factor=factor)
        configs_dicts.append(configs_dict)

    return configs_dicts


if __name__ == '__main__':
    bench = Benchmark(data_dir='../../../research_questions/survival_analysis/data/01_raw/six_datasets_lw.json')
    configs_dicts = hyperband(problem=bench, eta=2, random_seed=0,
                              max_budget_per_model=len(bench.query('Fashion-MNIST', 'epoch', 0)) - 1,
                              min_budget_per_model=2, dataset_name='Fashion-MNIST', factor=-1)
    plot_grey_box_optimization(configs_dicts, min_budget_per_model=2)
