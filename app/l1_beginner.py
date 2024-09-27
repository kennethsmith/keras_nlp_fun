import keras_nlp

NUM_CLASSES = 2


def get_model(preset):
    classifier = keras_nlp.models.BertClassifier.from_preset(
        preset=preset,
        num_classes=NUM_CLASSES
    )
    return classifier
