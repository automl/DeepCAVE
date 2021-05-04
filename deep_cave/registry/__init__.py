from typing import Union, Dict, List

import onnx

from .abstract_registry import AbstractRegistry
from .onnx_surrogate import ONNXSurrogate
from .onnx_registry import ONNXRegistry
from .onnx_surrogate import ONNXSurrogate


def log_surrogate(save_location: str, model_id: str, model: onnx.ModelProto):
    """
    Logs an onnx model.

    Allows to save surrogate model for later analysis. The model has to be converted into an onnx.ModelProto by the
    user.
    The model can be restored completely from this representation without any additional dependencies or side effects.
    The only requirement is having onnx_runtime installed.

    Parameters
    ----------
    save_location: str
        A valid directory to save the model to.
    model_id: str
        The model_id logged in Storage
    model: onnx.ModelProto
        The model proto that is saved to DeepCAVE.

    Returns
    -------
    None.
    """
    ONNXRegistry(save_location=save_location).log_surrogate_model(model=model, model_id=model_id)


def get_surrogate(save_location: str, model_id: str, mapping: Dict[str, List[str]]) -> ONNXSurrogate:
    """
    Load the saved model from log_surrogate and return a child class of AbstractSurrogate.

    If a sklearn model was logger, it can't be restored completely. The AbstractSurrogate is the proxy for that.
    It allows only to run inference with a DataFrame dataset.

    Parameters
    ----------
    save_location: str
        A valid directory to save the model to.
    model_id: str
        The model_id logged in Storage
    mapping: Dict
        This mapping defines which columns of the DataFrame dataset to use when running inference with the
        AbstractSurrogate. It also mapps the column names to the onnx input names defined in the onnx.ModelProto

    Returns
    -------
    AbstractSurrogate
    """
    return ONNXRegistry(save_location=save_location).get_surrogate(model_id=model_id, mapping=mapping)
