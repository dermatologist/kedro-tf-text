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