from more_itertools import take
import numpy as np
import os
import sys

import deep_cave


"""
Test the API interface and if it behaves as expected
"""


def test_workflow(tmpdir):
    study_name = 'test_workflow'
    deep_cave.set_study(study_name)
    deep_cave.set_tracking_uri(tmpdir.strpath)

    # explicitly call this to make sure a new and fresh study is initialized
    deep_cave.start_study()

    run_workflow()

    assert os.path.exists(tmpdir.join(study_name, 'trials.json'))

    # explicitly call this so the next test can initialize a new study
    deep_cave.end_study()
    assert os.path.exists(tmpdir.join(study_name, 'meta.json'))


def test_api_behaviour(tmpdir):
    api_behaviour(tmpdir.strpath)


def api_behaviour(uri):
    # test if wrong usage leads to defects
    deep_cave.end_study()

    # test if running multiple workflows in one run leads to problems
    deep_cave.set_tracking_uri(uri)
    deep_cave.set_study('test_workflow1')
    deep_cave.start_study()
    run_workflow(n_configs=10, num_params=2)
    deep_cave.end_study()

    deep_cave.set_study('test_workflow2')
    deep_cave.start_study()
    run_workflow(n_configs=10, num_params=2)
    deep_cave.end_study()

    # check if not ending a study leads to problems
    deep_cave.set_study('test_workflow3')
    deep_cave.start_study()
    run_workflow(n_configs=10, num_params=2)


def run_workflow(n_configs=100, num_params=5, min_budget=1):
    param_names = ['param_' + str(i) for i in range(num_params)]
    configs = np.random.randint(0, 2, (n_configs, num_params), dtype=np.int)
    num_selected = n_configs
    configs_to_eval = list(range(n_configs))
    config_dict = {config_id: {} for config_id in configs_to_eval}

    while int(num_selected) > 0:
        for id in configs_to_eval:
            with deep_cave.start_trial(dict(zip(param_names, configs[id])), min_budget) as trial:
                # generate some result
                result = np.random.random()
                config_dict[id][min_budget] = result
                trial.log_metric('objective', result)

        eval_configs_curr_budget = dict(filter(lambda x: min_budget in x[1], config_dict.items()))
        configs_to_eval = dict(sorted(eval_configs_curr_budget.items(), key=lambda x: x[1][min_budget]))
        configs_to_eval = list(take(num_selected, configs_to_eval.keys()))
        num_selected //= 2
        min_budget *= 2


if __name__ == '__main__':
    uri = os.path.join(os.path.join(os.path.dirname(__file__), '../studies/'))

    deep_cave.set_tracking_uri(uri)

    deep_cave.set_study('test_workflow')
    deep_cave.start_study()

    run_workflow(n_configs=10, num_params=2)

    print('break')