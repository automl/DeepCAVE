from functools import partial
from absl import app as flags_app
from absl import flags
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from smac.initial_design.random_configuration_design import RandomConfigurations
from deepcave import Recorder

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Seed', lower_bound=0)
flags.DEFINE_integer('n_configs', 500, '#Configurations', lower_bound=10)
flags.DEFINE_integer('task_id', 167149, 'Task Id', lower_bound=0)


def eval_func(cfg, seed=0, benchmark=None):
    result_dict = benchmark.objective_function(
        configuration=cfg, rng=seed, fidelity={"n_estimators": 64})
    cost = result_dict['cost']

    return cost


def main(argv):

    seed = FLAGS.seed
    task_id = FLAGS.task_id
    n_configs = FLAGS.n_configs

    benchmark = RandomForestBenchmark(task_id=task_id)
    cs = benchmark.get_configuration_space(seed=seed)

    tae = partial(eval_func, benchmark=benchmark)

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

    with Recorder(cs, prefix=f"rf_openml_{task_id}_smac_{seed}", overwrite=True) as r:
        for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in rh.data.items():
            config = rh.ids_config[config_id]
            if very_start_time is None:
                very_start_time = starttime

            starttime = starttime - very_start_time
            endtime = endtime - very_start_time

            r.start(config, start_time=starttime)
            r.end(costs=cost, end_time=endtime)

    #-------- START RANDOM SEARCH --------#
    with Recorder(cs, prefix=f"rf_openml_{task_id}_rs_{seed}", overwrite=True) as r:
        for config in cs.sample_configuration(n_configs):
            r.start(config)
            cost = tae(config, seed)
            r.end(costs=cost)


if __name__ == "__main__":
    flags_app.run(main)
