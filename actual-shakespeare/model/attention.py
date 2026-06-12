import math
import torch.nn as nn
import torch
from . import constants
from . import shared
import sys

class AttentionHead():
    def __init__(self):
        self.query_matrix = torch.randn((constants.FEATURE_DIMS, constants.QUERY_SIZE), device=shared.device)
        self.key_matrix = torch.randn(self.query_matrix.shape, device=shared.device)
        self.value_matrix_down = torch.randn((constants.FEATURE_DIMS, constants.QUERY_SIZE), device=shared.device)
        self.params = [self.query_matrix, self.key_matrix, self.value_matrix_down]
        for param in self.params:
            param *= shared.XavierFactor(param)

    def attend(self, feature_vectors):
        shared.check_eq(feature_vectors.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        if constants.DEBUG: print("ATTENTION")
        query_vectors = feature_vectors @ self.query_matrix
        shared.check_eq(query_vectors.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.QUERY_SIZE])
        key_vectors = feature_vectors @ self.key_matrix
        attention_matrix = query_vectors @ key_vectors.transpose(-2, -1)
        if constants.DEBUG: print(query_vectors.shape, key_vectors.shape, attention_matrix.shape)
        attention_matrix /= math.sqrt(constants.QUERY_SIZE)
        causal_mask = torch.triu(torch.ones_like(attention_matrix, dtype=torch.bool), diagonal=1)
        attention_matrix = attention_matrix.masked_fill(causal_mask, -math.inf)
        attention_matrix = torch.nn.functional.softmax(attention_matrix, dim=-1)
        attention_matrix = torch.nn.functional.dropout(attention_matrix, p=constants.DROPOUT, training=True)
        value_vectors = feature_vectors @ self.value_matrix_down
        enrichment_stack = attention_matrix @ value_vectors
        if constants.DEBUG: print(feature_vectors.shape, enrichment_stack.shape)
        return enrichment_stack
