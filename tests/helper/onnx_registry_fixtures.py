import pytest
import os

from sklearn.metrics import r2_score
from skl2onnx import to_onnx

from deep_cave.registry.onnx_registry import ONNXRegistry
from deep_cave.registry.onnx_surrogate import ONNXSurrogate
from deep_cave.util.util import get_random_string

from tests.helper.onnx_registry_requirements import get_features, get_dataset, get_model, get_objective


@pytest.fixture(scope='module')
def features():
    # filter out values that don't change.
    return get_features()


@pytest.fixture(scope='module')
def dataset():
    return get_dataset()


@pytest.fixture(scope='module')
def objective():
    return get_objective()


@pytest.fixture(scope='module')
def model(features, dataset, objective):
    return get_model(features, dataset, objective)


@pytest.fixture(scope='module')
def onnx_model(model, dataset, features):
    return to_onnx(model, dataset[features][:1].values)


@pytest.fixture(scope='module')
def onnx_input_name():
    # is the default from to_onnx, so hard code it here
    return 'X'


@pytest.fixture(scope='module')
def mapping(features, onnx_input_name):
    return {onnx_input_name: features}


@pytest.fixture(scope='module')
def model_prediction(dataset, model, features):
    return model.predict(dataset[features])


@pytest.fixture(scope='module')
def model_r2_score(model_prediction, dataset, objective):
    return r2_score(dataset[objective], model_prediction)


@pytest.fixture
def surrogate(tmpdir, onnx_model, mapping):
    model_id = get_random_string(15)
    # test if ONNXRegistry saves the model
    ONNXRegistry(save_location=tmpdir.strpath).log_surrogate_model(onnx_model, model_id=model_id)
    assert os.path.exists(os.path.join(tmpdir, model_id + '.onnx'))
    # if the file exists continue with loading the file and performing a prediction on some data
    surrogate = ONNXRegistry(save_location=tmpdir.strpath).get_surrogate(model_id=model_id, mapping=mapping)
    assert isinstance(surrogate, ONNXSurrogate)
    return surrogate