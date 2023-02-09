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



from typing import Any, Dict

import tensorflow as tf


#### TF method
# https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
def get_tf_bert_model(bert_model: Any, parameters: Dict) -> tf.keras.Model:
    (encoder, preprocessor) = bert_model
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input_for_bert")
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]      # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    # Read the documentation of the BERT model to understand the output format
    # ! The answer below explains which output to use
    # https://stackoverflow.com/questions/71980457/how-to-pass-bert-embeddings-to-an-lstm-layer
    embedding_model = tf.keras.Model(text_input, sequence_output, name="bert_embedding")
    return embedding_model
