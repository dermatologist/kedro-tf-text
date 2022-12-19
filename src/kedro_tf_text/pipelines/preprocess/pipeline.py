"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.1
"""
# No pipelines registered

from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_text.pipelines.preprocess.nodes import create_glove_embeddings, pickle_processed_text, json_processed_text

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

    """_summary_: This is a embedding pipeline 'preprocess'
    call this modular pipeline in your main pipeline with the following code:
    https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html
    """
def create_embedding_pipeline(**kwargs) -> Pipeline:
    return pipeline([

                    # node(
                    #     pickle_processed_text,
                    #     [datasets.get("text_data", "text_data"),
                    #      datasets.get("parameters", "parameters")],
                    #     datasets.get("processed_text", "processed_text"),
                    #     name="pickle_processed_text"
                    # ),
                    node(
                        func=json_processed_text,
                        inputs=["text_data"],
                        parameters=["params:embeddings"],
                        outputs="vocab_json",
                        name="create_vocab"
                    ),
                    node(
                        func=create_glove_embeddings,
                        inputs=["pretrained_embedding", "vocab_json"],
                        parameters=["params:embeddings"],
                        outputs="glove_embedding",
                        name="create_glove_embeddings"
                    ),

                    ])

