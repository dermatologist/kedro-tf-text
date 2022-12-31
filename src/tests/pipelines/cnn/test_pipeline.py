"""
This is a boilerplate test file for pipeline 'tabular'
generated using Kedro 0.18.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
from kedro.extras.datasets.pickle import PickleDataSet
from kedro_tf_text.pipelines.cnn.nodes import create_cnn_model


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="kedro_tf_text",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()


    @pytest.mark.run(order=3)
    def test_cnn_model(self, project_context):
        glovepath = "data/06_models/glove-embedding.pkl"
        cnn_path = "data/07_model_output/cnn-model"
        glove_model = PickleDataSet(filepath=glovepath)
        cnn_model = TensorFlowModelDataset(filepath=cnn_path)
        reloaded = glove_model.load()
        conf_params = project_context.config_loader.get('**/cnn.yml')
        data = create_cnn_model(reloaded, conf_params['cnn_model'])
        cnn_model.save(data)
        assert data is not None
