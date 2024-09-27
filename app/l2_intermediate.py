import keras_nlp

EPOCHS = 1
NUM_CLASSES = 2


def get_model(preset, imdb_train, imdb_test):
    classifier = keras_nlp.models.BertClassifier.from_preset(
        preset=preset,
        num_classes=NUM_CLASSES,
    )
    classifier.fit(
        imdb_train,
        validation_data=imdb_test,
        epochs=EPOCHS,
    )

    return classifier
