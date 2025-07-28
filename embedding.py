# Design inspired by GLoVe
# Used AI to write a few lines and describing functions
# Also used ChatGPT to come up with the overall design and ideas,
# but most actual coding, including implementation was done by me
# or utilized lines from AI but at my own pace
from collections import defaultdict

import keras
import math
import scipy as sp
from keras import Sequential
from keras.layers import Embedding
from scipy.sparse import lil_array
from datasets import load_dataset

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed
)
import pickle
import gc
import re

import tensorflow as tf

CONTEXT_WINDOW = 5
EMBEDDING_DIM = 16
X_MAX = 100 # Hyperparameter for weighting function
ALPHA = 0.75 # Hyperparameter for weighting function

SPECIAL_TOKENS = {
    'padding': '<PAD>',  # Used to pad sequences to equal length for batching
    'capitalize': '<CAPS>',  # Marks that the next token should be capitalized
    'separator': '<SEP>',  # Denotes the end of a section
    'begin_output': '<BOS>',  # Indicates the beginning of the output sequence
    'end_output': '<EOS>',  # Indicates the end of the output sequence
    'no_space': '<NOS>',  # Marks no space between words
    'unknown': '<UNK>', # Denotes an unknown token
}

MAX_VOCAB_SIZE = 30000
DATASET_SAMPLE_SIZE = 3000000

def count_word_freq(tokens, word_freq):
    for token in tokens:
        word_freq[token] += 1
    del tokens

def default_int():
    return -1

def default_str():
    return '<UNK>'

def gen_vocab(word_freq):
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocabulary = set()
    token_integer_map = defaultdict(default_int)
    integer_token_map = defaultdict(default_str)
    integer_token_map[-1] = '<UNK>'
    index = 0
    for i in range(min(len(sorted_words), MAX_VOCAB_SIZE)):
        print(sorted_words[i][0])
        vocabulary.add(sorted_words[i][0])
        token_integer_map[sorted_words[i][0]] = index
        integer_token_map[index] = sorted_words[i][0]
        index += 1

    special_tokens = list(SPECIAL_TOKENS.values())
    for i in range(min(len(sorted_words), MAX_VOCAB_SIZE), min(len(sorted_words), MAX_VOCAB_SIZE) + len(SPECIAL_TOKENS)):
        if special_tokens[i - min(len(sorted_words), MAX_VOCAB_SIZE)] not in vocabulary:
            vocabulary.add(special_tokens[i - min(len(sorted_words), MAX_VOCAB_SIZE)])
            token_integer_map[special_tokens[i - min(len(sorted_words), MAX_VOCAB_SIZE)]] = index
            integer_token_map[index] = special_tokens[i - min(len(sorted_words), MAX_VOCAB_SIZE)]

    return vocabulary | set(SPECIAL_TOKENS.values()), {'int': token_integer_map, 'str': integer_token_map}

def tokenize(text, vocabulary=None):
    token_pattern = re.compile(r"[0-9]|\w+|[^\w\s]")
    tokens = []

    index = 0
    while(index < len(text)):
        next_match = token_pattern.search(text, index)
        if next_match is None:
            break
        else:
            if next_match.start() == index and index != 0:
                tokens.append('<NOS>')
            token = text[next_match.start():next_match.end()]
            if vocabulary is not None and token not in vocabulary:
                token = '<UNK>'
            if token[0].isupper():
                token = token.lower()
                tokens.append('<CAPS>')
            tokens.append(token)
            index = next_match.end()

    return tokens

def process_batch_coocurrence_matrix(batch, count):
    tokens = list(map(lambda x: token_map['int'][x], tokenize(batch.numpy()[0].decode('utf-8'), vocabulary)))
    cooccurrence_matrix = lil_array((len(vocabulary), len(vocabulary)), dtype=float)
    gen_cooccurrence_matrix(cooccurrence_matrix, tokens)
    if count % 1000 == 0:
        print("Batch", count, "co-occurrences counted", flush=True)
    return cooccurrence_matrix.tocsr()

def gen_cooccurrence_matrix(cooccurrence_matrix, tokens):
    """
    Generates the co-occurrence matrix for a list of tokens.

    This function calculates the frequency of co-occurrences between each token
    in the provided list and its surrounding context tokens, determined by the
    context radius. The co-occurrence matrix is represented as a nested dictionary.

    :param tokens: A list of tokens (strings) for which the co-occurrence matrix will be created.
    :return: A nested dictionary where keys are tokens and values are dictionaries
             mapping neighboring tokens to their frequency of co-occurrence.
    """

    token_len = len(tokens)

    # AI Assisted
    for i in range(token_len):
        for j in range(i + 1, min(i + CONTEXT_WINDOW + 1, token_len)):
            increment = 1.0 / (j - i)
            cooccurrence_matrix[tokens[i], tokens[j]] += increment
            cooccurrence_matrix[tokens[j], tokens[i]] += increment

def glove_loss(y_true, y_pred):
    return tf.math.minimum(tf.math.pow(y_true / X_MAX, ALPHA), 1) * (tf.square(y_pred - tf.math.log(y_true)))

def create_model(coocurrence_matrix):
    vocab_size = coocurrence_matrix.shape[0]
    word_input = keras.layers.Input(shape=(1,), name="word_input")
    context_input = keras.layers.Input(shape=(1,), name="context_input")

    word_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIM,
        name="word_embedding")(word_input)
    context_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIM,
        name="context_embedding")(context_input)

    word_bias = keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE,
                                       output_dim=1, name="word_bias")(word_input)
    context_bias = keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE,
                                          output_dim=1, name="context_bias")(context_input)
    dot_matrix = keras.layers.Dot(axes=1)([word_embeddings, context_embeddings])
    addition = keras.layers.Add()([dot_matrix, word_bias, context_bias])

    output = keras.layers.Dense(units=1)(addition)

    model = keras.Model(inputs=[word_input, context_input], outputs=output, name="token_embedding_model")
    model.compile(optimizer="adam", loss=glove_loss)

    model.summary()
    return model

# initialize all the variables
# batch out the dataset data
# for each batch, run tokenize and update the co-occurrence matrix
# then continue

word_freq = defaultdict(int)

dataset = load_dataset("wikipedia", "20220301.en")
tf_dataset = dataset['train'].to_tf_dataset(columns=['text'], batch_size=1, shuffle=True).prefetch(tf.data.AUTOTUNE)

count = 0
for batch in tf_dataset:
    count_word_freq(tokenize(batch.numpy()[0].decode('utf-8')), word_freq)
    if count % 1000 == 0:
        print("Batch", count, "Tokenized")
    count += 1
    # gc.collect()
    # if count > DATASET_SAMPLE_SIZE:
    #     break

print("Word frequencies counted")

with open('word_freq.pkl', 'wb') as file:
    pickle.dump(word_freq, file)


vocabulary, token_map = gen_vocab(word_freq)

del word_freq
gc.collect()

print(vocabulary)

with open('token_map.pkl', 'wb') as file:
    pickle.dump(token_map, file)
with open('vocabulary.pkl', 'wb') as file:
    pickle.dump(vocabulary, file)

with open('token_map.pkl', 'rb') as file:
    token_map = pickle.load(file)
with open('vocabulary.pkl', 'rb') as file:
    vocabulary = pickle.load(file)

print("Vocabulary generated")

cooccurrence_matrix = lil_array((len(vocabulary), len(vocabulary)), dtype=float)

dataset = load_dataset("wikipedia", "20220301.en", streaming=True)

def dataset_generator():
    for datum in dataset['train']:
        yield datum['text']

tf_dataset = tf.data.Dataset.from_generator(dataset_generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.string)).batch(1)

count = 0
max_workers = 8
with (ThreadPoolExecutor(max_workers=max_workers) as executor):
    futures = []
    for batch in tf_dataset:
        futures = {executor.submit(process_batch_coocurrence_matrix, batch, count): batch}
        count += 1

        if count % max_workers == 0:
            for future in as_completed(futures):
                try:
                    cooccurrence_matrix += future.result()
                except Exception as e:
                    print(f"Error processing batch: {e}")

    # Get the stragglers just in case it doesn't evenly work out
    for future in as_completed(futures):
        try:
            cooccurrence_matrix += future.result()
            print("Batch", count, "co-occurrences counted")
        except Exception as e:
            print(f"Error processing batch: {e}")

cooccurrence_matrix = cooccurrence_matrix.tocsr()
#
# count = 0
# for batch in tf_dataset:
#     tokens = list(map(lambda x: token_map['int'][x], tokenize(batch.numpy()[0].decode('utf-8'), vocabulary)))
#     gen_cooccurrence_matrix(cooccurrence_matrix, tokens)
#     print("Batch", count, "Co-Occurences counted")
#     count += 1
#     del tokens
#     # gc.collect()
#     # if count > DATASET_SAMPLE_SIZE:
#     #     break


with open('token_map.pkl', 'wb') as file:
    pickle.dump(token_map, file)
with open('cooccurrence_matrix.pkl', 'wb') as file:
    pickle.dump(cooccurrence_matrix, file)

print("Co-occurrence matrix generated")

model = create_model(cooccurrence_matrix)

batch_size = 64

# Generated by AI
# Extract row, column, and data
rows, cols, data = [], [], []
for row_idx, (cols_in_row, data_in_row) in enumerate(zip(cooccurrence_matrix.rows, cooccurrence_matrix.data)):
    rows.extend([row_idx] * len(cols_in_row))
    cols.extend(cols_in_row)
    data.extend(data_in_row)

# Create TensorFlow SparseTensor
word_input = tf.convert_to_tensor(rows, dtype=tf.int64)
context_input = tf.convert_to_tensor(cols, dtype=tf.int64)
values = tf.convert_to_tensor(data, dtype=tf.float32)
dense_shape = tf.convert_to_tensor(cooccurrence_matrix.shape, dtype=tf.int64)

dataset = (
    tf.data.Dataset.from_tensor_slices(((word_input, context_input), values))
    .shuffle(buffer_size=len(data))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

print(dataset)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=3),
    keras.callbacks.ModelCheckpoint(filepath='embedding_model.h5', save_best_only=True)
]

history = model.fit(
    dataset,
    epochs=50,
    callbacks=callbacks,
    verbose=1,
)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
