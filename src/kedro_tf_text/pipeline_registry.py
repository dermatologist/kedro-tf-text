"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from kedro_tf_text.pipelines.preprocess.pipeline import create_glove_embedding_pipeline, pickle_processed_text_pipeline, create_bert_pipeline, create_preprocess_bert_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": create_glove_embedding_pipeline(),
    "bert": create_bert_pipeline(),
            "preprocess_bert": create_preprocess_bert_pipeline()}
