
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from deep_cave.runs.run import Run
import ConfigSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, Constant, UniformFloatHyperparameter

from smac.runhistory.runhistory import RunHistory
from ConfigSpace.read_and_write import json as cs_json



if __name__ == "__main__":
    with open('tests/configspace.json', 'r') as f:
        jason_string = f.read()
        cs = cs_json.read(jason_string)


    blub = ConfigSpace.ConfigurationSpace()

    a = CategoricalHyperparameter("test", choices=["a", "b"])
    blub.add_hyperparameter(a)

    b = UniformFloatHyperparameter("test2", lower=1, upper=5, log=True)
    blub.add_hyperparameter(b)

    c = CategoricalHyperparameter("test3", choices=["bbb", "ccc", "ccdc", "ccsc", "ccac", "ccec"])
    blub.add_hyperparameter(c)

    cond = ConfigSpace.EqualsCondition(b, a, "b")
    blub.add_condition(cond)

    cond = ConfigSpace.EqualsCondition(c, a, "b")
    blub.add_condition(cond)

    print(blub.sample_configuration().get_array())
    exit()

    rh = RunHistory()

    run = Run({}, rh, cs)
    new_cs = run.get_configspace(hyperparameter_ids=[3, 4])

    #for hp in new_cs.get_hyperparameters():
    #    print(hp)
    #    print(new_cs.get_children_of(hp))
    #    print("-")

    print(new_cs.sample_configuration())

    #print(cs)
    #print("---")
    #print(new_cs)




