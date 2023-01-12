""" Kedro pipeline nodes for preprocessing text data

@author: Bell Eapen
@date: 2021-05-01

Impliments:
    1. Convert a word2vec model to glove embeddings

"""

import re
from nltk.corpus import stopwords
import pandas as pd
from typing import Dict
from keras.layers import Embedding
import string
from gensim.models import Word2Vec
import numpy as np
import logging

TAG_RE = re.compile(r'<[^>]+>')

def clean_medical(text_list, max_seq_len=1000):
    text_list = [single_string.lower().strip() for single_string in text_list] # lower case & whitespace removal
    text_list = [re.sub(r'\d+', '', single_string) for single_string in text_list] # remove numerics
    text_list = [single_string.translate(str.maketrans("","",string.punctuation)) for single_string in text_list] # remove punctuation
    text_list = [tokenize(single_string) for single_string in text_list]
    text_list = [TAG_RE.sub('', single_string) for single_string in text_list] # remove html tags
    text_list = [single_string.replace('  ', ' ') for single_string in text_list] # remove double spaces
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


## * NODE
def create_word2vec(csv_data:pd.DataFrame, parameters: Dict):
    sentences = _process_csv_text(csv_data, parameters)
    model = Word2Vec(sentences, min_count=1) ## TODO add parameters
    return model # glove saves model as a pickle file

## * NODE
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

# Before embedding
def _process_csv_text(csv_data:pd.DataFrame, parameters: Dict):
    clean_data = clean_medical(csv_data[parameters['REPORT_FIELD']].tolist(), parameters['MAX_SEQ_LENGTH'])
    sentences = [line.lower().split(' ') for line in clean_data]
    padded_sentences = [sentence + [''] * (parameters['MAX_SEQ_LENGTH'] - len(sentence)) for sentence in sentences]
    return padded_sentences

# * NODE
def process_csv_text(csv_data:pd.DataFrame, model, parameters: Dict):
    sentences = _process_csv_text(csv_data, parameters)
    ids = csv_data[parameters['ID_FIELD']].tolist()
    # Encode the documents using the new embedding
    vocab = model.wv.key_to_index
    logging.info(f"Vocab size: {len(vocab)}")
    # encoded_docs = [[model.wv[word] for word in sentence] for sentence in sentences]
    encoded_docs = [[vocab[word] for word in sentence] for sentence in sentences]
    return_dict = {}
    for(id, doc) in zip(ids, encoded_docs):
        return_dict[id] = doc
    return return_dict
