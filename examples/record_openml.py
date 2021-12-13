from functools import partial
from absl import app as flags_app
from absl import flags
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from smac.initial_design.random_configuration_design import RandomConfigurations
from deepcave import Recorder, Objective

FLAGS = flags.FLAGS
flags.DEFINE_string('model', "nn", 'Model')
flags.DEFINE_integer('seed', 0, 'Seed', lower_bound=0)
flags.DEFINE_integer('n_configs', 100, '#Configurations', lower_bound=10)
flags.DEFINE_integer('task_id', 1779, 'Task Id', lower_bound=0)  # 167149
# 1768


def eval_func(cfg, seed=0, benchmark=None, fidelity={}):
    result_dict = benchmark.objective_function(
        configuration=cfg, rng=seed, fidelity=fidelity)

    cost = 1 - result_dict['info']['val_scores']['acc']

    return cost


def main(argv):

    model = FLAGS.model
    seed = FLAGS.seed
    task_id = FLAGS.task_id
    n_configs = FLAGS.n_configs
    objective = Objective("cost", lower=0, upper=1)

    if model == "rf":
        benchmark = RandomForestBenchmark(task_id=task_id)
        cs = benchmark.get_configuration_space(seed=seed)

        fidelity = {"n_estimators": 64}
        tae = partial(eval_func, benchmark=benchmark, fidelity=fidelity)
    elif model == "nn":
        benchmark = NNBenchmark(task_id=task_id)
        cs = benchmark.get_configuration_space(seed=seed)

        fidelity = {"iter": 50}
        tae = partial(eval_func, benchmark=benchmark, fidelity=fidelity)
    else:
        raise RuntimeError("Unknown model: {}".format(FLAGS.model))

    #-------- START RANDOM SEARCH --------#
    with Recorder(cs, objectives=objective, save_path="logs/JJ", prefix=f"{model}_openml_{task_id}_rs_{seed}", overwrite=True) as r:
        for config in cs.sample_configuration(n_configs):
            r.start(config)
            cost = tae(config, seed)
            r.end(costs=cost)

    #-------- START SMAC --------#
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        # Max number of function evaluations (the more the better)
        "runcount-limit": n_configs,
        "cs": cs,
        "deterministic": True,
    })

    smac = SMAC4HPO(
        scenario=scenario,
        tae_runner=tae,
        initial_design=RandomConfigurations,
        rng=seed)
    smac.optimize()

    rh = smac.get_runhistory()
    very_start_time = None

    with Recorder(cs, objectives=objective, save_path="logs/JJ", prefix=f"{model}_openml_{task_id}_smac_{seed}", overwrite=True) as r:
        for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in rh.data.items():
            config = rh.ids_config[config_id]
            if very_start_time is None:
                very_start_time = starttime

            starttime = starttime - very_start_time
            endtime = endtime - very_start_time

            r.start(config, start_time=starttime)
            r.end(costs=cost, end_time=endtime)


if __name__ == "__main__":
    flags_app.run(main)
