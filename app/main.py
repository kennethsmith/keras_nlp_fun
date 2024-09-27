import time
import os

import keras

import l1_beginner as l1
import l2_intermediate as l2
import l3_advanced as l3
import l4_expert as l4

# https://keras.io/guides/keras_nlp/getting_started/

BATCH_SIZE = 16

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

# Use mixed precision to speed up all training in this guide.
keras.mixed_precision.set_global_policy("mixed_float16")

# From: https://keras.io/api/keras_nlp/models/bert/bert_classifier/#frompreset-method
preset_model_names = [
    "bert_tiny_en_uncased",
    "bert_small_en_uncased",
    "bert_medium_en_uncased",
    "bert_base_en_uncased",
    "bert_base_en",
    "bert_base_zh",
    "bert_base_multi",
    "bert_large_en_uncased",
    "bert_large_en",
    # "bert_tiny_en_uncased_sst2",
]

data_paths = [
    '../data/aclImdb/train',
    '../data/aclImdb/test',
]


def load_data(path_train_data, path_test_data):
    imdb_train = keras.utils.text_dataset_from_directory(path_train_data, batch_size=BATCH_SIZE,)
    imdb_test = keras.utils.text_dataset_from_directory(path_test_data, batch_size=BATCH_SIZE)

    # Inspect first review
    # Format is (review text tensor, label tensor)
    print(imdb_train.unbatch().take(1).get_single_element())
    print(imdb_test.unbatch().take(1).get_single_element())

    return imdb_train, imdb_test


def main():
    print('Hello from ' + os.getcwd())
    preset = preset_model_names[0]
    imdb_train, imdb_test = load_data(data_paths[0], data_paths[1])

    time_and_execute('Beginner: ', l1.get_model(preset))
    time_and_execute('Intermediate: ', l2.get_model(preset, imdb_train, imdb_test))
    time_and_execute('Advanced Standard: ', l3.run_standard(preset, imdb_train, imdb_test))
    time_and_execute('Advanced Custom: ', l3.run_custom(preset, imdb_train, imdb_test))
    time_and_execute('Expert Standard: ', l4.run_standard(preset, imdb_train, imdb_test))
    time_and_execute('Expert Custom: ', l4.run_custom(imdb_train, imdb_test))


def query(c):
    q = "I love modular workflows in keras-nlp!"
    a = c.predict([q])
    print(q + ' result: ' + str(a))


def time_and_execute(n, f):
    s_time = time.time()
    model = f
    e_time = time.time()
    print(n + str(e_time - s_time))
    # query(model)


if __name__ == '__main__':
    main()
