"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


from kedro.pipeline import Pipeline, node, pipeline

from kedro_tf_text.pipelines.preprocess.nodes import (
    create_word2vec,
    gensim_to_keras_embedding,
    process_csv_text,
)

process_text_pipeline = Pipeline([
    node(
        func=process_csv_text, # returns a keras dataset as pickle file with the processed text
        # csv file ID | Long line of text or report. type pandas.CSVDataSet
        inputs=["text_data", "word2vec_embedding", "params:embedding"],
        outputs="processed_text",  # pickle.PickleDataSet
        name="pickle_processed_text",
        tags=["preprocess"]
    ),
])

glove_embedding = Pipeline([
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


def pickle_processed_text_pipeline(**kwargs) -> Pipeline:
    return process_text_pipeline

def create_glove_embedding_pipeline(**kwargs) -> Pipeline:
    return glove_embedding