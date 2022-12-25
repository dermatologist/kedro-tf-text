"""
This is a boilerplate pipeline 'bert'
generated using Kedro 0.18.4
"""

from typing import Any, Dict
import tensorflow as tf



#### TF method
# https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
def get_tf_bert_model(bert_model: Any, parameters: Dict) -> tf.keras.Model:
    (encoder, preprocessor) = bert_model
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]      # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    # Read the documentation of the BERT model to understand the output format
    # ! The answer below explains which output to use
    # https://stackoverflow.com/questions/71980457/how-to-pass-bert-embeddings-to-an-lstm-layer
    embedding_model = tf.keras.Model(text_input, sequence_output)
    return embedding_model
