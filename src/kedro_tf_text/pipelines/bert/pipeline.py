"""
This is a boilerplate pipeline 'bert'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from kedro_tf_text.pipelines.bert.nodes import get_tf_bert_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

download_bert = Pipeline([
    node(
        func=get_tf_bert_model, # downloads the bert model
        # kedro_tf_text.extras.datasets.bert_model_download.BertModelDownload
        # * see the catalogue for usage
        inputs=["bert_model", "params:bert_model"],
        # * tensorflow.TensorFlowModelDataset
        outputs="bert_model_saved",
        name="get_tf_bert_model",
        tags=["bert"]
    ),
])

def create_bert_pipeline(**kwargs) -> Pipeline:
    return download_bert

