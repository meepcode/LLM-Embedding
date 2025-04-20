# Design inspired by GLoVe
# Used AI to write a few lines and describing functions
# Also used ChatGPT to come up with the overall design and ideas,
# but most actual coding, including implementation was done by me
# or utilized lines from AI but at my own pace
from scipy.sparse import lil_array

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

CONTEXT_RADIUS = 5


def gen_cooccurrence_matrix(tokens, vocab_size):
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
