import sys  # noqa
sys.path.insert(0, '../')  # noqa

from absl import app as flags_app
from absl import flags
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from smac.initial_design.random_configuration_design import RandomConfigurations
from src import Recorder

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Seed', lower_bound=0)

task_id = 167149
n_configs = 500
benchmark = NNBenchmark(task_id=task_id)


def eval_func(cfg, seed=0):
    result_dict = benchmark.objective_function(
        configuration=cfg, rng=seed, fidelity={"iter": 100})
    cost = result_dict['cost']

    return cost


def main(argv):
    seed = FLAGS.seed
    cs = benchmark.get_configuration_space(seed=seed)

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
        tae_runner=eval_func,
        initial_design=RandomConfigurations,
        rng=seed)
    smac.optimize()

    rh = smac.get_runhistory()
    very_start_time = None

    with Recorder(cs, prefix=f"openml_{task_id}_smac_{seed}", overwrite=True) as r:
        for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in rh.data.items():
            config = rh.ids_config[config_id]
            if very_start_time is None:
                very_start_time = starttime

            starttime = starttime - very_start_time
            endtime = endtime - very_start_time

            r.start(config, start_time=starttime)
            r.end(costs=cost, end_time=endtime)

    #-------- START RANDOM SEARCH --------#
    with Recorder(cs, prefix=f"openml_{task_id}_rs_{seed}", overwrite=True) as r:
        for config in cs.sample_configuration(n_configs):
            r.start(config)
            cost = eval_func(config, seed)
            r.end(costs=cost)


if __name__ == "__main__":
    flags_app.run(main)
