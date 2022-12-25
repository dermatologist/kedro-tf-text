"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.1
"""
from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_text.pipelines.preprocess.nodes import create_glove_embeddings, create_word2vec, gensim_to_keras_embedding, pickle_processed_text, json_processed_text


# ! Deprecated, Use word2vec_embedding_pipeline instead
glove_embedding = Pipeline([
    node(
        func=json_processed_text, # return a json file with the vocab
        # csv file ID | Long line of text or report. type pandas.CSVDataSet
        inputs=["text_data", "params:embedding"],
        # type: json.JSONDataSet
        outputs="vocab_json",
        name="create_vocab_json",
        tags=["preprocess"]
    ),
    node(
        func=create_glove_embeddings, # return the glove embedding
        # pretrained word2vec embedding as text.TextDataSet Example: embd1000.txt
        # vocab_json as json.JSONDataSet (previous node)
        inputs=["pretrained_embedding", "vocab_json", "params:embedding"],
        outputs="glove_embedding", #pickle.PickleDataSet
        name="create_glove_embeddings",
        tags=["preprocess"]
    ),
])


process_text_pipeline = Pipeline([
    node(
        func=pickle_processed_text, # returns a keras dataset as pickle file with the processed text
        # csv file ID | Long line of text or report. type pandas.CSVDataSet
        inputs=["text_data", "params:embedding"],
        outputs="processed_text", #pickle.PickleDataSet
        name="pickle_processed_text",
        tags=["preprocess"]
    ),
])

word2vec_embedding = Pipeline([
    node(
        func=create_word2vec, # return the word2vec embedding
        inputs=["text_data", "params:embedding"],
        outputs="word2vec_embedding", #pickle.PickleDataSet
        name="create_word2vec_embeddings",
        tags=["preprocess"]
    ),
    node(
        func=gensim_to_keras_embedding, # return the keras embedding
        inputs=["word2vec_embedding", "params:embedding"],
        outputs="glove_embedding", #pickle.PickleDataSet
        name="create_keras_embeddings",
        tags=["preprocess"]
    ),
])

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

"""_summary_: This is a embedding pipeline 'preprocess'
call this modular pipeline in your main pipeline with the following code:
https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html
"""

def create_glove_embedding_pipeline(**kwargs) -> Pipeline:
    return glove_embedding

def pickle_processed_text_pipeline(**kwargs) -> Pipeline:
    return process_text_pipeline

def create_word2vec_embedding_pipeline(**kwargs) -> Pipeline:
    return word2vec_embedding