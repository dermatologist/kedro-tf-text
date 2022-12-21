# def dt_tabular_model(csv_data:pd.DataFrame, parameters: Dict):
#     y = csv_data.pop(parameters['TARGET'])
#     csv_data.drop(parameters['DROP'], axis=1, inplace=True)
#     X = csv_data
#     conf = ModelConfig(
#         nets=DeepFM,  # same as `nets=['linear','dnn_nets','fm_nets']`
#         categorical_columns='auto',  # or categorical_columns=['x1', 'x2', 'x3', ...]
#         # can be `metrics=['RootMeanSquaredError']` for regression task
#         # metrics=['AUC', 'accuracy'],
#         auto_categorize=True,
#         auto_discrete=False,
#         embeddings_output_dim=20,
#         embedding_dropout=0,
#     )
#     dt = DeepTable(config=conf)
#     deepmodel, history = dt.fit(X, y)
#     # https://github.com/DataCanvasIO/DeepTables/blob/master/deeptables/models/deeptable.py
#     return deepmodel.model
