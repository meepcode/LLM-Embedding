# Design inspired by GLoVe
# Used AI to write a few lines and describing functions
# Also used ChatGPT to come up with the overall design and ideas,
# but most actual coding, including implementation was done by me
# or utilized lines from AI but at my own pace
import keras
import math
import scipy as sp
from keras import Sequential
from keras.layers import Embedding
from scipy.sparse import lil_array

import re

import tensorflow as tf

MAX_VOCAB_SIZE = 10000
CONTEXT_WINDOW = 2
EMBEDDING_DIM = 50
X_MAX = 100 # Hyperparameter for weighting function
ALPHA = 0.75 # Hyperparameter for weighting function

SPECIAL_TOKENS = {
    'padding': '<PAD>',  # Used to pad sequences to equal length for batching
    'capitalize': '<CAPS>',  # Marks that the next token should be capitalized
    'separator': '<SEP>',  # Denotes the end of a sentence or section
    'begin_output': '<BOS>',  # Indicates the beginning of the output sequence
    'end_output': '<EOS>',  # Indicates the end of the output sequence
    'no_space': '<NOS>',  # Marks no space between words
}

def tokenize(text):
    token_patterns = [r"\w+", r"[^\w\s]"]
    token_pattern = re.compile(r"\w+|[^\w\s]")
    tokens = []

    index = 0
    while(index < len(text)):
        next_match = token_pattern.search(text, index)
        if next_match is None:
            break
        else:
            # if next_match.start() == index and index != 0:
                # tokens.append('<NOS>')
            token = text[next_match.start():next_match.end()]
            if token[0].isupper():
                token = token.lower()
                # tokens.append('<CAPS>')
            tokens.append(token)
            index = next_match.end()

    # tokens.insert(0, '<CLS>')
    # tokens.append('<SEP>')

    token_integer_map = {}
    integer_token_map = {}
    vocabulary = set(tokens)
    # vocabulary = vocabulary | set(SPECIAL_TOKENS.values())
    for index, token in enumerate(vocabulary):
        token_integer_map[token] = index
        integer_token_map[index] = token

    for index, token in enumerate(tokens):
        tokens[index] = token_integer_map[token]

    return tokens, vocabulary, (token_integer_map, integer_token_map)

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
    increment = 1

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

# sentence = "The concept of resilience, both on an individual and societal level, is a testament to the human spirit's remarkable ability to adapt, overcome, and thrive in the face of adversity. It embodies the capacity to bounce back from challenges, whether they stem from personal struggles, professional setbacks, or global crises such as pandemics, economic downturns, or natural disasters. Resilience is not merely an inherent trait but a skill that can be cultivated through self-awareness, practice, and support systems. On a personal level, resilience involves a combination of mental toughness, emotional regulation, and the ability to reframe negative experiences as opportunities for growth. Socially, resilience is bolstered by communities coming together, sharing resources, and fostering a culture of empathy and mutual aid. It requires individuals to recognize their interconnectedness and to act not only in their self-interest but also for the collective good. Moreover, in a world that is increasingly interconnected and fast-paced, resilience takes on new dimensions, requiring innovative thinking, adaptability to technology, and the ability to navigate cultural diversity with sensitivity and respect. Ultimately, resilience is a cornerstone of progress, allowing people to not only endure hardships but also emerge stronger, more compassionate, and better equipped to handle future challenges."
# sentence += " The bright sun shone brightly over the bright meadow, where bright flowers brightened the day. Birds chirped cheerfully, their cheerful notes creating a cheerful melody that filled the cheerful air. In the distance, children laughed joyfully, their joyful games adding joyful energy to the joyful scene. The joyful breeze carried the joyful scent of freshly cut grass, blending with the joyful aroma of blooming flowers. Everything felt perfectly bright and cheerful in the cheerful world under the cheerful sun."

sentence = "The cat sat on the mat"

tokens, vocabulary, tokenization_map = tokenize(sentence)
print("Sentence tokenized.")
print()

cooccurrence_matrix = lil_array((MAX_VOCAB_SIZE, MAX_VOCAB_SIZE))
gen_cooccurrence_matrix(cooccurrence_matrix, tokens)
print("Co-occurrence matrix generated.")
print()

for i in range(len(vocabulary)):
    for j in range(len(vocabulary)):
        if cooccurrence_matrix[i, j] != 0:
            print(tokenization_map[1][i], tokenization_map[1][j], cooccurrence_matrix[i, j])

def glove_loss(y_true, y_pred):
    return tf.math.minimum(tf.math.pow(y_true / X_MAX, ALPHA), 1) * (tf.square(y_pred - tf.math.log(y_true)))

def gen_embedding(coocurrence_matrix):
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

    # model.summary()

gen_embedding(cooccurrence_matrix)