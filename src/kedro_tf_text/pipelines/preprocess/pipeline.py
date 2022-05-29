"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.1
"""
# No pipelines registered

from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_text.pipelines.preprocess.nodes import create_glove_embeddings, pickle_processed_text, json_processed_text

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_embedding_pipeline(datasets={}, **kwargs) -> Pipeline:
    return pipeline([

                    # node(
                    #     pickle_processed_text,
                    #     [datasets.get("text_data", "text_data"),
                    #      datasets.get("parameters", "parameters")],
                    #     datasets.get("processed_text", "processed_text"),
                    #     name="pickle_processed_text"
                    # ),
                    node(
                        json_processed_text,
                        [datasets.get("text_data", "text_data"),
                         datasets.get("parameters", "parameters")],
                        datasets.get("vocab_json", "vocab_json"),
                        name="create_vocab"
                    ),
                    node(
                        create_glove_embeddings,
                        [datasets.get("pretrained_embedding", "pretrained_embedding"),
                         datasets.get("vocab_json", "vocab_json"), datasets.get("parameters", "parameters")],
                        datasets.get("glove_embedding", "glove_embedding"),
                        name="create_glove_embeddings"
                    ),

                    ])
