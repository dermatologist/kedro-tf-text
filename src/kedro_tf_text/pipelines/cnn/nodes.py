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

from keras import regularizers
from keras.layers import (
    Activation,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    concatenate,
)
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model

# https://github.com/faikaydin/medical-multimodal-with-transfer-learning/blob/master/cnn_model.py


def create_channel(x, filter_size, feature_map):
    """
    Creates a layer working channel wise
    """
    x = Conv1D(feature_map, kernel_size=filter_size, activation='relu', strides=1,
               padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    x = Flatten()(x)
    return x


def create_cnn_model(embedding_layer, parameters: Dict):

    #TODO:  Params
    num_words = None
    embedding_dim = None
    filter_sizes = [3, 4, 5]
    feature_maps = [100, 100, 100]
    max_seq_length = parameters['MAX_SEQ_LENGTH']
    dropout_rate = None
    multi = False

    if len(filter_sizes) != len(feature_maps):
        raise Exception(
            'Please define `filter_sizes` and `feature_maps` with the same length.')
    if not embedding_layer and (not num_words or not embedding_dim):
        raise Exception(
            'Please define `num_words` and `embedding_dim` if you not use a pre-trained embedding')

    if embedding_layer is None:
        embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                    input_length=max_seq_length,
                                    weights=None,
                                    trainable=True
                                    )

    channels = []
    x_in = Input(shape=(max_seq_length,), dtype='int32', name='cnn_text_input')
    emb_layer = embedding_layer(x_in)
    if dropout_rate:
        emb_layer = Dropout(dropout_rate)(emb_layer)
    for ix in range(len(filter_sizes)):
        x = create_channel(emb_layer, filter_sizes[ix], feature_maps[ix])
        channels.append(x)

    # Concatenate all channels
    x = concatenate(channels)
    concat = concatenate(channels)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = Activation('relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    if multi:
        return concat
    return Model(inputs=x_in, outputs=x)
