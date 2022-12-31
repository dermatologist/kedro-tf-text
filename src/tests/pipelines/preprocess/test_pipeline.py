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
from kedro.extras.datasets.pickle import PickleDataSet
from kedro_tf_text.pipelines.preprocess.nodes import create_word2vec, gensim_to_keras_embedding

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


    @pytest.mark.order1
    def test_w2v_model(self, project_context):
        csvpath = "data/01_raw/test_report.csv"
        w2vpath = "data/06_models/word2vec-embedding.pkl"
        data_set = CSVDataSet(filepath=csvpath)
        w2v_model = PickleDataSet(filepath=w2vpath)
        reloaded = data_set.load()
        conf_params = project_context.config_loader.get('**/preprocess.yml')
        data = create_word2vec(reloaded, conf_params['embedding'])
        w2v_model.save(data)
        assert data is not None


    @pytest.mark.order2
    def test_glove_model(self, project_context):
        w2vpath = "data/06_models/word2vec-embedding.pkl"
        glovepath = "data/06_models/glove-embedding.pkl"
        data_set = PickleDataSet(filepath=w2vpath)
        glove_model = PickleDataSet(filepath=glovepath)
        reloaded = data_set.load()
        conf_params = project_context.config_loader.get('**/preprocess.yml')
        data = gensim_to_keras_embedding(reloaded, conf_params['embedding'])
        glove_model.save(data)
        assert data is not None
