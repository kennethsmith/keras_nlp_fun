import keras_nlp
import keras
import tensorflow as tf

NUM_CLASSES = 2
SEQUENCE_LENGTH = 64
EPOCHS = 1


def get_preprocessor(preset):
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        preset,
        sequence_length=SEQUENCE_LENGTH,
    )
    return preprocessor


def get_preprocessor_custom(preset):
    tokenizer = keras_nlp.models.BertTokenizer.from_preset(preset)
    tokenizer(["I love modular workflows!", "Libraries over frameworks!"])

    # Write your own packer or use one of our `Layers`
    packer = keras_nlp.layers.MultiSegmentPacker(
        start_value=tokenizer.cls_token_id,
        end_value=tokenizer.sep_token_id,
        # Note: This cannot be longer than the preset's `sequence_length`, and there
        # is no check for a custom preprocessor!
        sequence_length=SEQUENCE_LENGTH,
    )

    def preprocessor(x, y):
        token_ids, segment_ids = packer(tokenizer(x))
        x = {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "padding_mask": token_ids != 0,
        }
        return x, y
    return preprocessor


def preprocess(imdb_train, imdb_test, preprocessor):
    imdb_train_preprocessed = imdb_train.map(preprocessor, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    imdb_test_preprocessed = imdb_test.map(preprocessor, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    print(imdb_train_preprocessed.unbatch().take(1).get_single_element())
    print(imdb_test_preprocessed.unbatch().take(1).get_single_element())

    return imdb_train_preprocessed, imdb_test_preprocessed


def preprocess_cached(imdb_train, imdb_test, preprocessor):
    imdb_train_cached = (imdb_train.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE))
    imdb_test_cached = (imdb_test.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE))

    print(imdb_train_cached.unbatch().take(1).get_single_element())
    print(imdb_test_cached.unbatch().take(1).get_single_element())

    return imdb_train_cached, imdb_test_cached


def get_model_custom(preset, train_data, test_data):
    backbone = keras_nlp.models.BertBackbone.from_preset(preset)
    backbone.trainable = False
    inputs = backbone.input
    sequence = backbone(inputs)["sequence_output"]
    for _ in range(2):
        sequence = keras_nlp.layers.TransformerEncoder(
            num_heads=2,
            intermediate_dim=64,
            dropout=0.1,
        )(sequence)
    # Use [CLS] token output to classify
    outputs = keras.layers.Dense(2)(sequence[:, backbone.cls_token_index, :])

    model = keras.Model(inputs, outputs)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(5e-5),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        jit_compile=True,
    )
    model.summary()
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
    )
    return model


def get_model(preset, train_data, test_data):
    classifier = keras_nlp.models.BertClassifier.from_preset(
        preset,
        preprocessor=None,
        num_classes=NUM_CLASSES
    )
    classifier.summary()
    classifier.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
    )
    return classifier


def run_standard(preset, imdb_train, imdb_test):
    preprocessor = get_preprocessor(preset)
    train_data, test_data = preprocess(imdb_train, imdb_test, preprocessor)

    model = get_model(preset, train_data, test_data)
    return model


def run_custom(preset, imdb_train, imdb_test):
    preprocessor = get_preprocessor_custom(preset)
    train_data, test_data = preprocess_cached(imdb_train, imdb_test, preprocessor)

    model = get_model_custom(preset, train_data, test_data)
    return model
