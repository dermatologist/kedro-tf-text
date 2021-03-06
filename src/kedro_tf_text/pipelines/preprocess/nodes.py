"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.1
"""

import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from keras.layers import Embedding
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.models.deepnets import DeepFM
import tensorflow as tf
from tensorflow.keras import layers
import string

def clean_medical(text_list):
    text_list = [single_string.lower().strip() for single_string in text_list] # lower case & whitespace removal
    text_list = [re.sub(r'\d+', '', single_string) for single_string in text_list] # remove numerics
    text_list = [single_string.translate(str.maketrans("","",string.punctuation)) for single_string in text_list] # remove punctuation
    text_list = [tokenize(single_string) for single_string in text_list]
    return text_list

def tokenize(doc):
    tokens = doc.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    return ' '.join(tokens)

def list_to_seq(text_list, num_words, seq_len):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)
    padded_sequences = pad_sequences(sequences, maxlen=seq_len, padding='post')
    return padded_sequences,tokenizer.word_index

def _process_csv(csv_data:pd.DataFrame, parameters: Dict):

    MAX_NUM_WORDS = parameters['MAX_NUM_WORDS']  # 15000
    EMBEDDING_DIM = parameters['EMBEDDING_DIM']  # 300
    MAX_SEQ_LENGTH = parameters['MAX_SEQ_LENGTH']  # 140

    clean_data = clean_medical(csv_data[parameters['REPORT_FIELD']].tolist())
    csv_data[parameters['REPORT_FIELD']] = clean_data
    # on average 40 words per document, keeping it a bit more then that
    seq_data, vocab = list_to_seq(
        text_list=clean_data, num_words=MAX_NUM_WORDS, seq_len=MAX_SEQ_LENGTH)
    return (seq_data, vocab)


def pickle_processed_text(csv_data:pd.DataFrame, parameters: Dict):
    (seq_data, vocab) = _process_csv(csv_data, parameters)
    return dict(zip(list(csv_data[parameters['ID_FIELD']]), seq_data))

def json_processed_text(csv_data:pd.DataFrame, parameters: Dict):
    (seq_data, vocab) = _process_csv(csv_data, parameters)
    return vocab


def create_glove_embeddings(load_from_text_dataset: str, load_vocab_from_json: Dict, parameters: Dict) -> Dict:

    # EMBEDDING
    """
        # in pipeline definition
        node(
            func=create_glove_embeddings,
            inputs=["text_dataset", "json_dataset", "params:embeddings"],
            outputs="pickle_dataset",
            name="unique node name"
        )
    """
    MAX_NUM_WORDS = parameters['MAX_NUM_WORDS']  # 15000
    EMBEDDING_DIM = parameters['EMBEDDING_DIM'] #300
    MAX_SEQ_LENGTH = parameters['MAX_SEQ_LENGTH'] #140

    embeddings_index = {}
    for line in load_from_text_dataset[1:].splitlines():
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    for word, i in load_vocab_from_json.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM,
                     input_length=MAX_SEQ_LENGTH,
                     weights=[embedding_matrix],
                     trainable=True
                     )

def dt_tabular_model(csv_data:pd.DataFrame, parameters: Dict):
    y = csv_data.pop(parameters['TARGET'])
    csv_data.drop(parameters['DROP'], axis=1, inplace=True)
    X = csv_data
    conf = ModelConfig(
        nets=DeepFM,  # same as `nets=['linear','dnn_nets','fm_nets']`
        categorical_columns='auto',  # or categorical_columns=['x1', 'x2', 'x3', ...]
        # can be `metrics=['RootMeanSquaredError']` for regression task
        # metrics=['AUC', 'accuracy'],
        auto_categorize=True,
        auto_discrete=False,
        embeddings_output_dim=20,
        embedding_dropout=0,
    )
    dt = DeepTable(config=conf)
    deepmodel, history = dt.fit(X, y)
    # https://github.com/DataCanvasIO/DeepTables/blob/master/deeptables/models/deeptable.py
    return deepmodel.model

########## https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


def get_vocab_size(tokenizer):
    # calculate vocabulary size
    return len(tokenizer.word_index) + 1
##########


def tabular_model(csv_data: pd.DataFrame, parameters: Dict):
    """Builds a DNN based on tabular data and returns the model.
    Ref: https://www.tensorflow.org/tutorials/load_data/csv

    Args:
        csv_data (pd.DataFrame): _description_
        parameters (Dict): _description_

    Returns:
        _type_: _description_
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
        layers.Dense(64),
        layers.Dense(1)
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

    model.fit(x=csv_features_dict, y=csv_labels, epochs=10)
    return model
