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



from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers


def tabular_model(csv_data: pd.DataFrame, parameters: Dict):
    """Builds a DNN based on tabular data and returns the model.
    Ref: https://www.tensorflow.org/tutorials/load_data/csv

    Args:
        csv_data (pd.DataFrame): _description_
        parameters (Dict): _description_

    Returns:
        _type_: A compiled Keras model of tabular data
    """

    csv_features = csv_data.copy()

    csv_features.drop(parameters['DROP'], axis=1, inplace=True)
    csv_features.pop(parameters['TARGET'])

    inputs = {}

    for name, column in csv_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


    numeric_inputs = {name: input for name, input in inputs.items()
                    if input.dtype == tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(csv_features[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)

    preprocessed_inputs = [all_numeric_inputs]


    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        lookup = layers.StringLookup(vocabulary=np.unique(csv_features[name]))
        one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)


    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    csv_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    return csv_model(csv_preprocessing, csv_data, inputs, parameters)

def csv_model(preprocessing_head, csv_data, inputs, parameters: Dict):
    """_summary_

    Args:
        preprocessing_head (_type_): _description_
        csv_data (_type_): _description_
        inputs (_type_): _description_
        parameters (Dict): _description_

    Returns:
        _type_: _description_
    """

    csv_features = csv_data.copy()
    body = tf.keras.Sequential([
        layers.Dense(parameters['DENSE_LAYER']),
        layers.Dense(1)  # binary classification will be removed during fusion
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=tf.optimizers.Adam())

    csv_labels = csv_features.pop(parameters['TARGET'])


    csv_features.drop(parameters['DROP'], axis=1, inplace=True)

    csv_features_dict = {name: np.array(value)
                                for name, value in csv_features.items()}

    model.fit(x=csv_features_dict, y=csv_labels, epochs=parameters['EPOCHS'])
    return model


