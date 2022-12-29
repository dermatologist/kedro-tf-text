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




########################

# https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
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


"""_summary_

From a csv file with an ID field and report field, extracts the sequence of tokens and the vocabulary
seq_data can be pickled and used as input to the model

"""


def pickle_processed_text(csv_data: pd.DataFrame, parameters: Dict):
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


def json_processed_text(csv_data: pd.DataFrame, parameters: Dict):
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
    EMBEDDING_DIM = parameters['EMBEDDING_DIM']  # 300
    MAX_SEQ_LENGTH = parameters['MAX_SEQ_LENGTH']  # 140

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


# https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
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
