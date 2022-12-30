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

from kedro.extras.datasets.text import TextDataSet
from kedro.extras.datasets.json import JSONDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
from kedro_tf_text.extras.datasets.bert_model_download import BertModelDownload

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

    def test_bert_download(self, project_context):
        dataset = {
            "type": "kedro_tf_text.extras.datasets.bert_model_download.BertModelDownload",
            "preprocessor_url": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "encoder_url": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        }
        preprocessor_url = dataset['preprocessor_url']
        encoder_url = dataset['encoder_url']
        data_set = BertModelDownload(preprocessor_url=preprocessor_url, encoder_url=encoder_url)
        data = data_set.load()
        assert data is not None
