"""
This is a boilerplate pipeline 'tabular'
generated using Kedro 0.18.4
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