"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.1
"""

import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd

def clean_medical(text_list):
    text_list = [single_string.lower().strip() for single_string in text_list] # lower case & whitespace removal
    text_list = [re.sub(r'\d+', '', single_string) for single_string in text_list] # remove numerics
    text_list = [single_string.translate(str.maketrans("","",single_string.punctuation)) for single_string in text_list] # remove punctuation
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


def pickle_processed_text(csv_data:pd.DataFrame):
    """_summary_

    Args:
        csv_data (pd.DataFrame): ID - file name of the corresponding image & Text - report of the image


    Returns:
        _type_: _description_
    """
    clean_data = clean_medical(list(csv_data.Text))
    csv_data['Text'] = clean_data
    seq_data, vocab = list_to_seq(text_list=clean_data, num_words=15000, seq_len=140) # on average 40 words per document, keeping it a bit more then that
    return dict(zip(list(csv_data['ID']),seq_data))

def json_processed_text(csv_data:pd.DataFrame):
    clean_data = clean_medical(list(csv_data.Text))
    csv_data['Text'] = clean_data
    seq_data, vocab = list_to_seq(text_list=clean_data, num_words=15000, seq_len=140) # on average 40 words per document, keeping it a bit more then that
    return vocab