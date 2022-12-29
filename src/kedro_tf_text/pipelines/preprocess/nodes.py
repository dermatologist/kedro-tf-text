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
from gensim.models import Word2Vec

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

# def list_to_seq(text_list, num_words, seq_len):
#     tokenizer = Tokenizer(num_words=num_words)
#     tokenizer.fit_on_texts(text_list)
#     sequences = tokenizer.texts_to_sequences(text_list)
#     padded_sequences = pad_sequences(sequences, maxlen=seq_len, padding='post')
#     return padded_sequences,tokenizer.word_index


## NODE
def create_word2vec(csv_data:pd.DataFrame, parameters: Dict):
    sentences = _process_csv_text(csv_data, parameters)
    model = Word2Vec(sentences, min_count=3)
    return model # glove saves model as a pickle file

## NODE
def gensim_to_keras_embedding(model, parameters: Dict):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

    Parameters
    ----------
    train_embeddings : bool
        If False, the returned weights are frozen and stopped from being updated.
        If True, the weights can / will be further updated in Keras.

    Returns
    -------
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    """
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array
    # which row in `weights` corresponds to which word?
    index_to_key = keyed_vectors.index_to_key

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=parameters['train_embedding'],
    )
    return layer

def _process_csv_text(csv_data:pd.DataFrame, parameters: Dict):
    clean_data = clean_medical(csv_data[parameters['REPORT_FIELD']].tolist())
    sentences = [line.lower().split(' ') for line in clean_data]
    return sentences

def process_csv_text(csv_data:pd.DataFrame, parameters: Dict):
    sentences = _process_csv_text(csv_data, parameters)
    sentences = [list(sentences)]
    return sentences
