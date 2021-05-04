from typing import Dict, List

import numpy as np
import onnxruntime as rt
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from ..util.logs import get_logger

logger = get_logger(__name__)


class ONNXSurrogate(BaseEstimator, RegressorMixin):
    """
    Wrapper model to query onnx inference session with a dataframe.

    The Class automatically maps the columns to the correct and expected input.

    Currently, only designed for sklearn models and pipelines.
    """
    def __init__(self, model: rt.InferenceSession, mapping: Dict[str, List[str]] = None):
        """
        Wrapper model for sklearn compatibility. The ONNX runtime session has defined inputs and outputs.
        The predict method will get the trial data from store and perform a prediction.

        Since, we can't know from the model, which of the trial data was used for training and in what order
        the class also requires a mapping dict. The dict contains the mapping from the model input onto
        the feature names used.

        This not only solves the ordering problem, it also allows the model to be trained only on a subset of
        the available features. E.g. when some of the features never change, they can be ommitted in training.
        Additionally, sklearn.pipelines can used, when the preprocessing of features is necesary. For example
        nominal data has to be transformed into one hot encoding. There is no way to tell the server, the preprocessing
        steps. Instead, the model saved is a sklearn pipeline, that does the preprocessing.
        But when only a subset of the features is nominal it should be converted. Defining multiple entries
        into the pipeline solves this selection problem. ONNX handles this by having multiple input names.
        For each input the correct features have to be used.

        Parameters
        ----------
        model
            onnxruntime.InferenceSession. Provided by the ONNXRegistry, which handles the loading and saving
        mapping
            Dict. For each input in the ONNX model, the corresponding list contains the feature names, that can be
                found in the trial data config columns.
        """
        self.model = model
        self.mapping = mapping
        # do some sanity checks
        for expected_input in self.model.get_inputs():
            features = self.mapping[expected_input.name]
            if len(features) != expected_input.shape[1]:
                logger.exception(f'Size of features from model.mapping {len(features)} doesn\'t fit the '
                                 f'expected input shape {expected_input.shape[1]}')
                raise ValueError(f'Size of features from model.mapping {len(features)} doesn\'t fit the '
                                 f'expected input shape {expected_input.shape[1]}')
        self.fit()

    def predict(self, features_df: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        For each expected input for the model, the correct features are selected, based on the information given in
        mapping. The inputs are also cast to the expected data type.

        Then the model infers a prediction from the input. Currently only one output is allowed.
        Staying compatible with most sklearn methods.

        Parameters
        ----------
        features_df
            trials DataFrame from store.
        kwargs
            Kwargs. Used for compatibility with sklearn.

        Returns
        -------

        """
        onnx_input = {}
        for expected_input in self.model.get_inputs():
            # construct the input for onnx. Under the assumption mapping, features_df and onnx_input fit
            onnx_input[expected_input.name] = features_df[self.mapping[expected_input.name]].values.astype(
                self._onnx_type_to_dtype(expected_input.type))
        # run the inference
        # todo rework, so that only the actual required output gets returned
        # currently this is sufficient for RF
        return self.model.run([output.name for output in self.model.get_outputs()], onnx_input)[0]

    @staticmethod
    def _onnx_type_to_dtype(onnx_type: str):
        """
        Helper function to convert cryptic types of onnx to actual numpy dtypes.

        Parameters
        ----------
        onnx_type: str
            return value from onnx_model.get_outputs()
        Returns
        -------
            A valid numpy dtype
        """
        if onnx_type == 'tensor(double)':
            return np.float64
        if onnx_type == 'tensor(float)':
            return np.float32
        else:
            logger.warning(f'{onnx_type} currently not supported')
            raise ValueError(f'{onnx_type} currently not supported')

    def fit(self, x=None, y=None, **kwargs):
        """
        Here for compatibility. The model is already trained when loaded. Later modifications aren't possible.
        """
        self.is_fitted_ = True
        return self