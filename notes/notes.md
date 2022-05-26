    last_layer = deepmodel.model.layers[-2].output
    text_last_layer = BatchNormalization()(last_layer)
    return text_last_layer