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

