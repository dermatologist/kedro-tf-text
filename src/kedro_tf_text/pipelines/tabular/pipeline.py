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

from kedro_tf_text.pipelines.tabular.nodes import tabular_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


tabular_model_pipeline = Pipeline([
    node(
        func=tabular_model, # returns a keras model of the tabular data
        # tabular_data as pandas.CSVDataSet with ## ID | included | fields | excluded | fields | outcome (y)
        inputs=["tabular_data", "params:tabular"],
        outputs="tabular_model", #pickle.PickleDataSet
        name="create_tabular_model",
        tags=["tabular"]
    ),
])


def create_tabular_model_pipeline(**kwargs) -> Pipeline:
    return tabular_model_pipeline