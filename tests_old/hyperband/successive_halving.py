import numpy as np

import sys
from os.path import dirname, abspath

from more_itertools import take

sys.path.append(dirname(dirname(abspath(__file__))))

from libs.LCBench.api import Benchmark
from tests.hyperband.utils import plot_grey_box_optimization  # noqa: E402

import deep_cave


def successive_halving(problem, n_models, budgets, eta, random_seed, dataset_name,
                       factor=1):
    """
    The successive halving algorithm, called as subroutine in hyperband.
    :param problem: An instance of problem
    :param n_models: int;  The number of configs to evaluate
    :param min_budget_per_model: int
    :param max_budget_per_model: int
    :param eta: float
    :param random_seed: int
    :return:
    """
    np.random.seed(random_seed)
    configs_dict = {i: {'config': np.random.randint(0, problem.get_number_of_configs(dataset_name)),
                        'f_evals': {}} for i in range(n_models)}

    configs_to_eval = list(range(n_models))
    num_configs_to_proceed = n_models
    for b in budgets:
        # Evaluate the configs selected for this budget
        for config_id in configs_to_eval:
            # ...
            # returns validation mse and runtime
            # save for each configuration at the number of resources (translated to number of epochs) what the
            # valid mse and runtime is
            with deep_cave.start_trial(problem.get_config(dataset_name, str(configs_dict[config_id]['config'])), b) as trial:
                result = problem.query(dataset_name, 'Train/val_accuracy', configs_dict[config_id]['config'])[b]
                configs_dict[config_id]['f_evals'][b] = result
                trial.log_metric('train_val_accuracy', result)

        # Compute number of configs to proceed to next higher budget
        # At each step half the number of models to proceed
        num_configs_to_proceed = num_configs_to_proceed // eta

        # Select the configs from the configs_dict which have been evaluated on the current budget b
        eval_configs_curr_budget = dict(filter(lambda x: b in x[1]['f_evals'], configs_dict.items()))

        # Out of these configs select the ones to proceed to the next higher budget and assign this
        # list to configs_to_eval
        configs_to_eval = dict(sorted(eval_configs_curr_budget.items(), key=lambda x: factor * x[1]['f_evals'][b]))
        configs_to_eval = dict(take(num_configs_to_proceed, configs_to_eval.items()))

        # Increase the budget for the next SH iteration.
        # After half of the models disappeared double the budget? In the lecture it was 4 times the resources
        # in http://ais.informatik.uni-freiburg.de/teaching/ws18/deep_learning_lab/presentation_automl.pdf
        # slide 26 it looks like two
        b = b * eta
    return configs_dict


if __name__ == '__main__':
    bench = Benchmark(data_dir='../../../research_questions/survival_analysis/data/01_raw/six_datasets_lw.json')
    configs_dict = successive_halving(problem=bench, n_models=20, eta=2, random_seed=0,
                                      budgets=[3, 6, 12, 25, 50], dataset_name='Fashion-MNIST', factor=-1)
    plot_grey_box_optimization([configs_dict], min_budget_per_model=10)
