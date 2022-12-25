""" Kedro pipeline nodes for preprocessing text data

@author: Bell Eapen
@date: 2021-05-01

Impliments:
    1. Convert a word2vec model to glove embeddings

"""

import sys
from absl import flags
import re
from nltk.corpus import stopwords
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from keras.layers import Embedding
import tensorflow as tf
from keras import layers
import string

TAG_RE = re.compile(r'<[^>]+>')

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

"""_summary_

From a csv file with an ID field and report field, extracts the sequence of tokens and the vocabulary
seq_data can be pickled and used as input to the model

"""
def pickle_processed_text(csv_data:pd.DataFrame, parameters: Dict):
    # TODO: rename this function to something more meaningful
    """_summary_

    Args:
        csv_data (pd.DataFrame): data with ID and report fields
        parameters (Dict): Kedro parameters

    Returns:
        Dict: returns a dictionary with ID as key and sequence of tokens as value
    """
    (seq_data, vocab) = _process_csv(csv_data, parameters)
    return dict(zip(list(csv_data[parameters['ID_FIELD']]), seq_data))


def json_processed_text(csv_data:pd.DataFrame, parameters: Dict):
    # TODO: rename this function to something more meaningful
    """_summary_

    Args:
        csv_data (pd.DataFrame): data with ID and report fields
        parameters (Dict): Kedro parameters

    Returns:
        Dict: vocabulary
    """
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

## Cleaning text functions for BERT ##
# Ref: https://github.com/artelab/Image-and-Text-fusion-for-UPMC-Food-101-using-BERT-and-CNNs/blob/main/BERT_LSTM.ipynb
# https://gist.github.com/dermatologist/062c46eafe8c118334a004f6cfab663d
def _preprocess_text(sen: str) -> str:
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    sentence = sentence.lower()

    return sentence

# https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
vec_preprocess_text = np.vectorize(_preprocess_text)

def remove_tags(text: str) -> str:
    return TAG_RE.sub('', text)

# Preprocessing of texts according to BERT


def _get_masks(text: str, max_length: int, tokenizer: Any):
    """Mask for padding"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]
    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))


vec_get_masks = np.vectorize(_get_masks, signature='(),(),()->(n)')


def _get_segments(text: str, max_length: int, tokenizer: Any):
    """Segments: 0 for the first sequence, 1 for the second"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    segments = []
    current_segment_id = 0
    with_tags = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))


vec_get_segments = np.vectorize(_get_segments, signature='(),(),()->(n)')


def _get_ids(text: str, max_length: int, tokenizer: Any):

    # TODO: Fix this https://github.com/google-research/bert/issues/1133
    sys.argv = ['preserve_unused_tokens=False']
    flags.FLAGS(sys.argv)

    """Token ids from Tokenizer vocab"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids


vec_get_ids = np.vectorize(_get_ids, signature='(),(),()->(n)')

def prepare_for_bert(text_array: List, tokenizer: Any, parameters: Dict):
    max_length = parameters['MAX_LENGTH']
    """Prepares text for BERT"""
    ids = vec_get_ids(text_array,
                      max_length,
                      tokenizer).squeeze()
    masks = vec_get_masks(text_array,
                          max_length,
                          tokenizer).squeeze()
    segments = vec_get_segments(text_array,
                                max_length,
                                tokenizer).squeeze()

    return ids, segments, masks

def preprocess_text_bert(data: pd.DataFrame, bert_model: Any,  parameters: Dict) -> pd.DataFrame:
    """Preprocesses text

    Args:
        data (pd.DataFrame): _description_
        parameters (Dict): _description_

    Returns:
        pd.DataFrame: _description_
    """

    (bert_layer, vocab_file, tokenizer) = bert_model
    text = parameters['REPORT_FIELD']
    max_length = parameters['MAX_LENGTH']
    processed_data = vec_preprocess_text(data[text].values)
    ids, segments, masks = prepare_for_bert(processed_data, tokenizer, parameters)
    return ids, segments, masks


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