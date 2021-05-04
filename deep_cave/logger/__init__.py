import datetime
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union, List

from ConfigSpace import ConfigurationSpace
import onnx

from ..util.logs import get_logger
from ..store.trial import Trial
from .. import store
from .. import registry

logger = get_logger(__name__)


# todo add a dataset field
def start_study(study: Optional[str] = None,
                optimizer_args: Optional[dict] = None,
                objective: Optional[str] = None,
                search_space: ConfigurationSpace = None,
                **groups) -> None:
    """
    Is automatically called and filled with default values when study is None.
    Otherwise, start_study can be used to specify more details about the current study.

    Parameters
    ----------
    study: str, optional
        Select a name of the study, when none generate a random string.
    optimizer_args: Dict, optional
        The arguments used in the AutoML optimizer to run this study. Should be any JSON serializable dict.
    objective: str, optional
        The objective the AutoML optimizer tries to optimized. Should be the subset of logged metrics
    search_space: ConfigurationSpace
        The configuration space of used by the AutoML optimizer. This is used for logging purposes and
         to generate new instance. Currently, only ConfigurationSpace is supported
    groups: dict
        Key-value pair that classifies the study into a group. E.g., {"dataset": "MNIST", "run": 5, "algorithm": BOHB}
    Returns
    -------
    None
        The global store of deep_cave is set to this study, but no value is returned.
    """
    store.start_study(**locals())


def log_surrogate(model: onnx.ModelProto, fidelity: Union[str, float] = None, mapping: Dict[str, List[str]] = None):
    """
    Log a surrogate model as onnx.ModelProto.

    Convert your surrogate model (with one of the available converters) into onnx representation.
    Create a mapping from the onnx input name to the used features in the correct ordering.

    The above arguments are important to use the surrogate model correctly. The model can then be used
    in the server application to generate Interpretability plots about the surrogate model.

    For more information see the supported tools section https://onnx.ai/supported-tools.html#buildModel or
    or the onnx tutorial page https://github.com/onnx/tutorials.

    Parameters
    ----------
    model: onnx.ModelProto
        The model returned from convert_* (e.g. convert_sklearn from the package skl2onnx)
    fidelity: str, float, optional
        Optional, to specify on which fidelity the model was build. If None, it is logged as the default model
    mapping
        A dictionary with the onnx input name as key. The value has to be a list of feature names, that are
        consistent with the one used in configs. The feature names are used, in case only a subset of the configurations
        where used to train the surrogate model. The ordering of the feature names is also important to guarantee,
        that the ordering corresponds to the one used during training.
        Additionally, can this mapping be used in more complex Pipelines (like sklearn.pipeline.Pipeline)
        or models with multiple inputs.

    Returns
    -------

    """
    # first register the model with storage to get the model_id
    model_id = store.log_surrogate(fidelity=fidelity, mapping=mapping)
    # then save the model
    registry.log_surrogate(save_location=store.get_registry_uri(), model_id=model_id, model=model)


def end_study() -> None:
    """
    Ends the current study and resets the global study store to None

    Returns
    -------
    None
    """
    store.end_study()


# for easier use: At this level of the API add usability functions like contextmanager


@contextmanager
def start_trial(config: Dict, fidelity: Union[str, float],
                increment: Optional[Union[datetime.datetime, datetime.date, datetime.time, int]] = None) -> Trial:
    """
    Starts a Trial. A Trial is uniquely identified by the config and its budget.
    Can be used as a context manager, which ends the trial automatically after exiting the scope.

    The Trial object can then be used as a thread safe way to log certain metrics to the trial.

    Parameters
    ----------
    config: Dict
        A dictionary that describes the hyperparameters used to train the ML system.
        The config is hashed to generate a unique key. Fidelity shouldn't alter the config.
        Otherwise, the trackability between trials with the same config and different
        fidelities are lost
    fidelity: [str, float]
        A fidelity that influences the behaviour of the ML system.
        It could be a float, describing the percentage of data used to train the system.
        It could be an int, specifying the budget in epochs available to train the ML system.
        It could be a str, describing the type of evaluation function used. E.g. evaluate
        the model in high or low fidelity.
    increment: [datetime.datetime, datetime.date, datetime.time, int], optional
        Can be any type of datetime object or int increment or seed. If None the current time is used.
        This parameter can be used to track a benchmark, where start and end time can be
        different from the wall clock time needed to produce the result.
        The increment or seed or start_time is used to make the tuple config, fidelity unique

    Returns
    -------
    Trial
        The Trial object can be used to log metrics for the running trial to the Study.
    """
    try:
        trial = store.start_trial(config=config, fidelity=fidelity, increment=increment)
        yield trial
    except Exception as e:
        import traceback
        logger.exception(traceback.extract_tb(e.__traceback__))
        raise e
    finally:
        end_trial(trial)


def end_trial(trial: Trial, end_time: Optional[Union[datetime.date, datetime.datetime, datetime.time]] = None):
    """
    Function to end a trial. Has an optional end_time argument when used with pre-calculated
    benchmarks, to correctly track the running time.

    end_trial is implicitly called, when using start_trial with a context manager.

    Parameters
    ----------
    trial: Trial
        The Trial object to reference the current trial.
    end_time: [datetime.date, datetime.datetime, datetime.time], optional
        A datetime object that is logged as end_time and which is used to calculate the
        running duration.
        If end_time is None, set it to the current wall clock time.

    Returns
    -------
    None
        Only alters the store of the storage.
    """
    store.end_trial(trial, end_time=end_time)
