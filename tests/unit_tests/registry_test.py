import pytest
import os

import numpy as np

from tests.helper.onnx_registry_fixtures import *
from tests.helper.onnx_registry_requirements import get_features, get_dataset, get_model, get_objective

from sklearn.metrics import r2_score


def test_surrogate(surrogate, dataset, model_prediction, model_r2_score, objective):
    # surrogate should be able to select only the relevant feature columns and then return a prediction
    prediction = surrogate.predict(dataset)
    # the prediction should be a relative well approximation of the training data
    assert np.allclose(r2_score(dataset[objective], prediction), model_r2_score)
    # the prediction should also be really close to the prediction of the original sklearn model
    assert np.allclose(prediction[:, 0], model_prediction)


def test_surrogate_sklearn_compatibility(surrogate, dataset):
    # assert that the surrogate model is accepted as BaseEstimator by sklearn
    from sklearn.inspection import partial_dependence
    assert partial_dependence(surrogate, dataset, [0])


def test_surrogate_pdp_compatibility(surrogate, dataset):
    from pdpbox.info_plots import actual_plot
    fig, axes, summary_df = actual_plot(surrogate, dataset,
                                        feature='learning_rate',
                                        feature_name='learning_rate',
                                        predict_kwds={})


if __name__ == '__main__':
    # makes the code executable without pytest, to generate the models files
    base_path = os.path.dirname(__file__)
    dataset = get_dataset()

    # filter out values that don't change.
    features = get_features()

    rf = get_model(features, dataset, get_objective())
    # convert it to ONNX representation. Infer the datatypes from the nd array
    onx = to_onnx(rf, dataset[features][:1].values)

    model_path = os.path.join(base_path, '../models/')
    ONNXRegistry(model_path).log_surrogate_model(onx, '../rf_surrogate')