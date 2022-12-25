"""
This is a boilerplate pipeline 'cnn'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from kedro_tf_text.pipelines.cnn.nodes import create_cnn_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


cnn_text_pipeline = pipeline([
    node(
        func=create_cnn_model,
        inputs=['glove_embedding','params:cnn_text_model'],
        outputs='cnn_text_model',
        name='create_cnn_model',
        tags=['cnn_text_model']
    )
])

def create_cnn_pipeline(**kwargs) -> Pipeline:
    return cnn_text_pipeline