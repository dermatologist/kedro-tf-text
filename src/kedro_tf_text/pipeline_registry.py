"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro_tf_text.pipelines.bert.pipeline import create_bert_pipeline

from kedro_tf_text.pipelines.preprocess.pipeline import create_glove_embedding_pipeline, pickle_processed_text_pipeline
from kedro_tf_text.pipelines.tabular.pipeline import create_tabular_model_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": pickle_processed_text_pipeline(),  
            "bert": create_bert_pipeline(),
            "glove": create_glove_embedding_pipeline(),
            "text": pickle_processed_text_pipeline(),
            "tabular": create_tabular_model_pipeline(),
    }

