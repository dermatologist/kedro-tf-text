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

from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from kedro_tf_text.pipelines.tabular.nodes import tabular_model


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

    def test_tabular_model(self, project_context):
        csvpath = "data/01_raw/test_dataset.csv"
        tfpath = "data/06_models/tabular_model"
        data_set = CSVDataSet(filepath=csvpath)
        save_args ={
            'save_format': 'tf'
        }
        tf_model = TensorFlowModelDataset(filepath=tfpath, save_args=save_args)
        reloaded = data_set.load()
        conf_params = project_context.config_loader.get('**/tabular.yml')
        data = tabular_model(reloaded, conf_params['tabular'])
        tf_model.save(data)
        assert data is not None
