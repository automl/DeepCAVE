from abc import ABC, abstractmethod
from typing import Dict, List

import onnx

from .onnx_surrogate import ONNXSurrogate


class AbstractRegistry(ABC):
    """
    Blueprint and interface for the implementation of ONNXRegistry.
    """

    def __init__(self, save_location: str):
        """
        Every Registry should have one location where all the models are saved.

        Parameters
        ----------
        save_location
            str. String identifying the save and load location of models.
        """
        self.save_location = save_location

    @property
    @abstractmethod
    def format(self) -> str:
        """
        Maybe useful later, when there is more than one serialization format for models.

        Returns
        -------
            Returns a string matching the format this class uses.
        """
        pass

    @abstractmethod
    def log_surrogate_model(self, model: onnx.ModelProto, model_id: str) -> None:
        """
        Interface for saving the actual model. Every model gets an unique id string, which links it to the meta
        data in model_id.

        Parameters
        ----------
        model
            onnx.ModelProto. The model converted into ONNX proto format.
        model_id
            Unique string provided by the system
        Returns
        -------
            Nothing.
        """
        pass

    @abstractmethod
    def get_surrogate(self, model_id: str, mapping: Dict[str, List[str]]) -> ONNXSurrogate:
        """
        Interface for retrieving the model based on its unique model id. Mapping and id come from Store.

        Parameters
        ----------
        model_id
            str. Unqiue model id.
        mapping
            Dict. Mapping between model input requirements and features in the trial data.
        Returns
        -------
            Return the surrogate as ONNXSurrogate model, which is compatible with sklearn.
        """
        pass
