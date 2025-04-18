# Design inspired by GLoVe

import torch.nn as nn
from functions import tokenize

CONTEXT_RADIUS = 5

def update_occurrence_matrix(occurrence_matrix, token, cotokens):
    for cotoken in cotokens:
        occurrence_matrix[token] = occurrence_matrix.get(token, {})
        occurrence_matrix[token][cotoken] = (
            occurrence_matrix[token].get(cotoken, 0) + 1)

def gen_occurrence_matrix(tokens):
    occurrence_matrix = {}

    for i in range(len(tokens)):
        for j in range(max(0, i - CONTEXT_RADIUS), min(i + CONTEXT_RADIUS + 1, len(tokens))):
            cotokens = set()
            if i != j:
                cotokens.add(tokens[j])

        update_occurrence_matrix(occurrence_matrix, tokens[i], cotokens)

    return occurrence_matrix

# sentence = "apple apple tree apple"
# tokens = tokenize(sentence)
# print(gen_occurrence_matrix(tokens))
