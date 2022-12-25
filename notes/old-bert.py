
TAG_RE = re.compile(r'<[^>]+>')
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
