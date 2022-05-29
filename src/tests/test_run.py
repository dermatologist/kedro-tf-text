"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from kedro.extras.datasets.text import TextDataSet
from kedro.extras.datasets.json import JSONDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
from kedro_tf_text.pipelines.preprocess.nodes import create_glove_embeddings, json_processed_text, pickle_processed_text, tabular_model


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


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

    # def test_glove_embeddings(self, project_context):
    #     filepath = "data/04_feature/new_data_embed300.txt"
    #     jsonpath = "data/04_feature/vocab.json"
    #     picklepath = "data/06_models/glove_embeddings.pickle"
    #     data_set = TextDataSet(filepath=filepath)
    #     json_data = JSONDataSet(filepath=jsonpath)
    #     pickle_data = PickleDataSet(filepath=picklepath)
    #     reloaded = data_set.load()
    #     jsonloaded = json_data.load()
    #     conf_params = project_context.config_loader.get('**/preprocess.yml')
    #     data = create_glove_embeddings(reloaded, jsonloaded, conf_params['embeddings'])
    #     pickle_data.save(data)
    #     assert data is not None

    def test_tabular_model(self, project_context):
        csvpath = "data/01_raw/test_dataset.csv"
        tfpath = "data/06_models/tabular_model"
        data_set = CSVDataSet(filepath=csvpath)
        save_args ={
            'save_format': 'tf'
        }
        tf_model = TensorFlowModelDataset(filepath=tfpath, save_args=save_args)
        reloaded = data_set.load()
        conf_params = project_context.config_loader.get('**/preprocess.yml')
        data = tabular_model(reloaded, conf_params)
        tf_model.save(data)
        assert data is not None

    def test_process_text(self, project_context):
        csvpath = "data/01_raw/test_report.csv"
        jsonpath = "data/02_intermediate/vocab.json"
        picklepath = "data/02_intermediate/text_model.pickle"
        data_set = CSVDataSet(filepath=csvpath)
        json_data_set = JSONDataSet(filepath=jsonpath)
        pickle_data = PickleDataSet(filepath=picklepath)
        reloaded = data_set.load()
        conf_params = project_context.config_loader.get('**/preprocess.yml')
        data = pickle_processed_text(reloaded, conf_params)
        json_data = json_processed_text(reloaded, conf_params)
        pickle_data.save(data)
        json_data_set.save(json_data)
        assert data is not None
