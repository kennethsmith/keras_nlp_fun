import keras_nlp
import keras
import tensorflow as tf

SEQUENCE_LENGTH = 64
EPOCHS = 1


def get_preprocessor_standard(preset):
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        preset,
        sequence_length=SEQUENCE_LENGTH,
    )
    return preprocessor


def get_masker(tokenizer):
    # keras.Layer to replace some input tokens with the "[MASK]" token
    masker = keras_nlp.layers.MaskedLMMaskGenerator(
        vocabulary_size=tokenizer.vocabulary_size(),
        mask_selection_rate=0.25,
        mask_selection_length=64,
        mask_token_id=tokenizer.token_to_id("[MASK]"),
        unselectable_token_ids=[
            tokenizer.token_to_id(x) for x in ["[CLS]", "[PAD]", "[SEP]"]
        ],
    )
    return masker


def get_preprocessor_custom_split(p, m):
    def preprocessor(inputs, label):
        inputs = p(inputs)
        masked_inputs = m(inputs["token_ids"])
        # Split the masking layer outputs into a (features, labels, and weights)
        # tuple that we can use with keras.Model.fit().
        features = {
            "token_ids": masked_inputs["token_ids"],
            "segment_ids": inputs["segment_ids"],
            "padding_mask": inputs["padding_mask"],
            "mask_positions": masked_inputs["mask_positions"],
        }
        labels = masked_inputs["mask_ids"]
        weights = masked_inputs["mask_weights"]
        return features, labels, weights
    return preprocessor


def preprocess(train_data, test_data, preprocessor):
    train_ds = train_data.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_data.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    print(train_ds.unbatch().take(1).get_single_element())
    print(test_ds.unbatch().take(1).get_single_element())

    return train_ds, test_ds


def get_model_custom(tokenizer, train_data, test_data):
    # BERT backbone
    backbone = keras_nlp.models.BertBackbone(
        vocabulary_size=tokenizer.vocabulary_size(),
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        intermediate_dim=64,
    )

    # Language modeling head
    mlm_head = keras_nlp.layers.MaskedLMHead(
        token_embedding=backbone.token_embedding,
    )

    inputs = {
        "token_ids": keras.Input(shape=(None,), dtype=tf.int32, name="token_ids"),
        "segment_ids": keras.Input(shape=(None,), dtype=tf.int32, name="segment_ids"),
        "padding_mask": keras.Input(shape=(None,), dtype=tf.int32, name="padding_mask"),
        "mask_positions": keras.Input(shape=(None,), dtype=tf.int32, name="mask_positions"),
    }

    # Encoded token sequence
    sequence = backbone(inputs)["sequence_output"]

    # Predict an output word for each masked input token.
    # We use the input token embedding to project from our encoded vectors to
    # vocabulary logits, which has been shown to improve training efficiency.
    outputs = mlm_head(sequence, mask_positions=inputs["mask_positions"])

    # Define and compile our pretraining model.
    model = keras.Model(inputs, outputs)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(learning_rate=5e-4),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
        jit_compile=True,
    )
    model.summary()
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
    )
    return model


def get_vocab(train_data):
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        train_data.map(lambda x, y: x),
        vocabulary_size=20_000,
        lowercase=True,
        strip_accents=True,
        reserved_tokens=["[PAD]", "[START]", "[END]", "[MASK]", "[UNK]"],
    )
    return vocab


def get_tokenizer(vocab):
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=True,
        strip_accents=True,
        oov_token="[UNK]",
    )
    return tokenizer


def get_packer(tokenizer):
    packer = keras_nlp.layers.StartEndPacker(
        start_value=tokenizer.token_to_id("[START]"),
        end_value=tokenizer.token_to_id("[END]"),
        pad_value=tokenizer.token_to_id("[PAD]"),
        sequence_length=512,
    )
    return packer


def get_preprocessor_custom(tokenizer, packer):
    def preprocessor(x, y):
        token_ids = packer(tokenizer(x))
        return token_ids, y

    return preprocessor


def get_inputs():
    inputs = keras.Input(
        shape=(None,),
        dtype="int32",
        name="token_ids",
    )
    return inputs


def get_outputs(token_id_input, vocab, packer):
    outputs = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=len(vocab),
        sequence_length=packer.sequence_length,
        embedding_dim=64,
    )(token_id_input)

    outputs = keras_nlp.layers.TransformerEncoder(
        num_heads=2,
        intermediate_dim=128,
        dropout=0.1,
    )(outputs)

    # Use "[START]" token to classify
    outputs = keras.layers.Dense(2)(outputs[:, 0, :])

    return outputs


def get_model(inputs, outputs, train_data, test_data):
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
    )
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


def run_standard(preset, imdb_train, imdb_test):
    preprocessor = get_preprocessor_standard(preset)
    tokenizer = preprocessor.tokenizer
    masker = get_masker(tokenizer)
    preprocessor_custom = get_preprocessor_custom_split(preprocessor, masker)
    train_data, test_data = preprocess(imdb_train, imdb_test, preprocessor_custom)

    model = get_model_custom(tokenizer, train_data, test_data)
    return model


def run_custom(imdb_train, imdb_test):
    vocab = get_vocab(imdb_train)
    tokenizer = get_tokenizer(vocab)
    packer = get_packer(tokenizer)
    preprocessor = get_preprocessor_custom(tokenizer, packer)
    inputs = get_inputs()
    outputs = get_outputs(inputs, vocab, packer)
    train_data, test_data = preprocess(imdb_train, imdb_test, preprocessor)

    model = get_model(inputs, outputs, train_data, test_data)
    return model
