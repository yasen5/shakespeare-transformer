import math
import torch.nn as nn
import torch
import constants
import shared

class AttentionHead():
    def __init__(self):
        self.query_matrix = torch.randn((constants.N_FEATURE_DIMS, constants.QUERY_SIZE), device=shared.device)
        self.key_matrix = torch.randn(self.query_matrix.shape, device=shared.device)
        self.value_matrix_up = torch.randn((constants.N_FEATURE_DIMS, constants.QUERY_SIZE), device=shared.device)
        self.value_matrix_down = torch.randn((constants.QUERY_SIZE, constants.N_FEATURE_DIMS), device=shared.device)
        self.params = [self.query_matrix, self.key_matrix, self.value_matrix_up, self.value_matrix_down]
        for param in self.params:
            param.requires_grad = True
        # self.attention_matrix = torch.empty((constants.CONTEXT_WINDOW_SIZE, constants.CONTEXT_WINDOW_SIZE))

    def attend(self, feature_vectors):
        shared.check_eq(feature_vectors.shape, [constants.CONTEXT_WINDOW_SIZE, constants.N_FEATURE_DIMS])
        if constants.DEBUG: print("ATTENTION")
        query_vectors = feature_vectors @ self.query_matrix  # TODO pytorch vectorize
        key_vectors = feature_vectors @ self.key_matrix # TODO pytorch vectorize
        attention_matrix = query_vectors @ key_vectors.t()
        if constants.DEBUG: print(query_vectors.shape, key_vectors.shape, attention_matrix.shape)
        
        attention_matrix /= math.sqrt(constants.QUERY_SIZE)
        if constants.DEBUG: print(attention_matrix)
        causal_mask = torch.triu(torch.ones_like(attention_matrix, dtype=torch.bool), diagonal=1)
        attention_matrix = attention_matrix.masked_fill(causal_mask, -math.inf)
        attention_matrix = torch.softmax(attention_matrix, dim=1)
        enrichment_stack = torch.zeros_like(feature_vectors)
        for row in range(attention_matrix.shape[0]):
            for col in range(attention_matrix.shape[1]):
                enrichment_stack[row] += attention_matrix[row,col] * self.value_matrix_up @ self.value_matrix_down @ feature_vectors[col]
        if constants.DEBUG: print(feature_vectors.shape, enrichment_stack.shape)
        return (feature_vectors + enrichment_stack).view(-1, constants.CONTEXT_WINDOW_SIZE, constants.N_FEATURE_DIMS)