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

import pickle

import re

import tensorflow as tf

MAX_VOCAB_SIZE = 30000
CONTEXT_WINDOW = 2
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

def count_word_freq(tokens, word_freq):
    for token in tokens:
        word_freq[token] += 1

def gen_vocab(word_freq):
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocabulary = set()
    token_integer_map = defaultdict(lambda: -1)
    integer_token_map = defaultdict(lambda: '<UNK>')
    i = 0
    while i < len(sorted_words):
        vocabulary.add(sorted_words[i][0])
        i += 1

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

    for i in range(min(token_len, CONTEXT_WINDOW)):
        for j in range(i + 1, min(token_len, CONTEXT_WINDOW + 1)):
            increment = 1.0 / (j - i)
            cooccurrence_matrix[tokens[i], tokens[j]] += increment
            cooccurrence_matrix[tokens[j], tokens[i]] += increment

    for i in range(min(token_len, CONTEXT_WINDOW), token_len - CONTEXT_WINDOW):
        for j in range(i + 1, min(token_len, i + CONTEXT_WINDOW + 1)):
            increment = 1.0 / (j - i)
            cooccurrence_matrix[tokens[i], tokens[j]] += increment
            cooccurrence_matrix[tokens[j], tokens[i]] += increment

    for i in range(token_len - CONTEXT_WINDOW, token_len):
        for j in range(i + 1, min(token_len, i + CONTEXT_WINDOW + 1)):
            increment = 1.0 / (j - i)
            cooccurrence_matrix[tokens[i], tokens[j]] += increment
            cooccurrence_matrix[tokens[j], tokens[i]] += increment

def glove_loss(y_true, y_pred):
    return tf.math.minimum(tf.math.pow(y_true / X_MAX, ALPHA), 1) * (tf.square(y_pred - tf.math.log(y_true)))

def create_model(coocurrence_matrix):
    vocab_size = min(coocurrence_matrix.shape[0], MAX_VOCAB_SIZE)
    word_input = keras.layers.Input(shape=(1,), name="Word Input")
    context_input = keras.layers.Input(shape=(1,), name="Context Input")

    word_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIM,
        name="Word Embedding")(word_input)
    context_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIM,
        name="Context Embedding")(context_input)

    word_bias = keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE,
                                       output_dim=1, name="Word Bias")(word_input)
    context_bias = keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE,
                                          output_dim=1, name="Context Bias")(context_input)
    dot_matrix = keras.layers.Dot(axes=1)([word_embeddings, context_embeddings])
    addition = keras.layers.Add()([dot_matrix, word_bias, context_bias])

    output = keras.layers.Dense(units=1)(addition)

    model = keras.Model(inputs=[word_input, context_input], outputs=output, name="Token Embedding Model")
    model.compile(optimizer="adam", loss=glove_loss)

    model.summary()
    return model

# initialize all the variables
# batch out the dataset data
# for each batch, run tokenize and update the co-occurrence matrix
# then continue

word_freq = defaultdict(int)

vocabulary = set(SPECIAL_TOKENS.values())
token_integer_map = defaultdict(lambda: -1)
integer_token_map = defaultdict(lambda: -1)
cooccurrence_matrix = lil_array((MAX_VOCAB_SIZE, MAX_VOCAB_SIZE), dtype=float)

dataset = load_dataset("wikipedia", "20220301.en")
tf_dataset = dataset['train'].to_tf_dataset(columns=['text'], batch_size=1).prefetch(tf.data.AUTOTUNE)

for batch in tf_dataset:
    print(batch.numpy()[0])
    count_word_freq(tokenize(batch.numpy()[0].decode('utf-8')), word_freq)

print("Word_freq counted")

vocabulary, token_map = gen_vocab(word_freq)

print("Vocabulary generated")

for batch in tf_dataset:
    tokens = map(lambda x: token_map['int'][x], tokenize(batch.numpy()[0].decode('utf-8')))
    gen_cooccurrence_matrix(cooccurrence_matrix, list(token_integer_map.values()))

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
for row_idx, (cols_in_row, data_in_row) in enumerate(zip(cooccurrence_matrix.rows, coo.data)):
    rows.extend([row_idx] * len(cols_in_row))
    cols.extend(cols_in_row)
    data.extend(data_in_row)

# Create TensorFlow SparseTensor
indices = tf.convert_to_tensor(list(zip(rows, cols)), dtype=tf.int64)
values = tf.convert_to_tensor(data, dtype=tf.float32)
dense_shape = tf.convert_to_tensor(lil_array.shape, dtype=tf.int64)

sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)

dataset = (
    tf.data.Dataset.from_tensor_slices(sparse_tensor)
    .shuffle(buffer_size=sparse_tensor.shape[0].numpy())
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
    epochs=20,
    callbacks=callbacks,
    verbose=1,
)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('history.pkl', 'wb') as file:
    pickle.dump(history.history, file)