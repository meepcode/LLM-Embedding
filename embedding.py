# Design inspired by GLoVe
# Used AI to write a few lines and describing functions
# Also used ChatGPT to come up with the overall design and ideas,
# but most actual coding, including implementation was done by me
# or utilized lines from AI but at my own pace
from scipy.sparse import lil_array

import re

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

CONTEXT_WINDOW = 2

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

def gen_cooccurrence_matrix(tokens, vocab_size):
    for index, token in enumerate(tokens):
        tokens[index] = token_integer_map[token]

    return tokens, vocabulary, (token_integer_map, integer_token_map)

    """
    Generates the co-occurrence matrix for a list of tokens.

    This function calculates the frequency of co-occurrences between each token
    in the provided list and its surrounding context tokens, determined by the
    context radius. The co-occurrence matrix is represented as a nested dictionary.

    :param tokens: A list of tokens (strings) for which the co-occurrence matrix will be created.
    :return: A nested dictionary where keys are tokens and values are dictionaries
             mapping neighboring tokens to their frequency of co-occurrence.
    """

    token_len = tf.shape(tokens)[0].numpy()

    cooccurrence_matrix = lil_array((vocab_size, vocab_size), dtype=int)

    for i in range(token_len):
        for j in range(max(0, i - CONTEXT_RADIUS),
                       min(i + CONTEXT_RADIUS + 1, token_len)):
            token = tokens[i].numpy()
            cotoken = tokens[j].numpy()

            if token != cotoken:
                cooccurrence_matrix[token, cotoken] += 1

    return cooccurrence_matrix

sentence = "The concept of resilience, both on an individual and societal level, is a testament to the human spirit's remarkable ability to adapt, overcome, and thrive in the face of adversity. It embodies the capacity to bounce back from challenges, whether they stem from personal struggles, professional setbacks, or global crises such as pandemics, economic downturns, or natural disasters. Resilience is not merely an inherent trait but a skill that can be cultivated through self-awareness, practice, and support systems. On a personal level, resilience involves a combination of mental toughness, emotional regulation, and the ability to reframe negative experiences as opportunities for growth. Socially, resilience is bolstered by communities coming together, sharing resources, and fostering a culture of empathy and mutual aid. It requires individuals to recognize their interconnectedness and to act not only in their self-interest but also for the collective good. Moreover, in a world that is increasingly interconnected and fast-paced, resilience takes on new dimensions, requiring innovative thinking, adaptability to technology, and the ability to navigate cultural diversity with sensitivity and respect. Ultimately, resilience is a cornerstone of progress, allowing people to not only endure hardships but also emerge stronger, more compassionate, and better equipped to handle future challenges."
sentence += " The bright sun shone brightly over the bright meadow, where bright flowers brightened the day. Birds chirped cheerfully, their cheerful notes creating a cheerful melody that filled the cheerful air. In the distance, children laughed joyfully, their joyful games adding joyful energy to the joyful scene. The joyful breeze carried the joyful scent of freshly cut grass, blending with the joyful aroma of blooming flowers. Everything felt perfectly bright and cheerful in the cheerful world under the cheerful sun."

vectorize_layer = TextVectorization(
    standardize='lower'
)

vectorize_layer.adapt(sentence)
cooccurence_matrix = gen_cooccurrence_matrix(vectorize_layer(sentence), vectorize_layer.vocabulary_size())

for i in range(vectorize_layer.vocabulary_size()):
    for j in range(vectorize_layer.vocabulary_size()):
        if cooccurence_matrix[i, j] > 2:
            print(vectorize_layer.get_vocabulary()[i], vectorize_layer.get_vocabulary()[j], cooccurence_matrix[i, j])
