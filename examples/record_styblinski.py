from functools import partial
from absl import app as flags_app
from absl import flags
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from smac.initial_design.random_configuration_design import RandomConfigurations
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from deepcave import Recorder, Objective

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Seed', lower_bound=0)
flags.DEFINE_integer('n_configs', 30, '#Configurations', lower_bound=10)

d = 5


def eval_func(cfg, seed=0):
    cost = 0
    for i in range(1, d+1):
        lamb = cfg[f"lamb_{i}"]
        cost += lamb**4 + 16*lamb**2 + 5*lamb

    cost = 1/2 * cost

    return cost


def main(argv):
    seed = FLAGS.seed
    n_configs = FLAGS.n_configs
    objective = Objective("cost", lower=0, upper=1)

    cs = CS.ConfigurationSpace(seed=seed)

    # Create configspace
    for i in range(1, d+1):
        lamb = CSH.UniformFloatHyperparameter(
            f"lamb_{i}", lower=-5, upper=5, log=False)
        cs.add_hyperparameter(lamb)

    #-------- START RANDOM SEARCH --------#
    with Recorder(cs, objectives=objective, save_path="logs/JJ", prefix=f"styblinski_rs_30_{seed}", overwrite=True) as r:
        for config in cs.sample_configuration(n_configs):
            r.start(config)
            cost = eval_func(config, seed)
            r.end(costs=cost)

    #-------- START RANDOM SEARCH --------#
    with Recorder(cs, objectives=objective, save_path="logs/JJ", prefix=f"styblinski_rs_300_{seed}", overwrite=True) as r:
        for config in cs.sample_configuration(n_configs*10):
            r.start(config)
            cost = eval_func(config, seed)
            r.end(costs=cost)

    exit()

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

    with Recorder(cs, objectives=objective, save_path="logs/JJ", prefix=f"styblinski_smac_{seed}", overwrite=True) as r:
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
