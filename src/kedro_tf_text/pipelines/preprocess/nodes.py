""" Kedro pipeline nodes for preprocessing text data

@author: Bell Eapen
@date: 2021-05-01

Impliments:
    1. Convert a word2vec model to glove embeddings

"""

import re
from nltk.corpus import stopwords
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from typing import Dict
from keras.layers import Embedding
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
