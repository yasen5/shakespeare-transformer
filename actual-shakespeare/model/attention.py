import math
import torch.nn as nn
import torch
from . import constants
from . import shared
import sys
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_matrix = nn.Linear(constants.FEATURE_DIMS, constants.QUERY_SIZE, bias=False)
        self.key_matrix = nn.Linear(constants.FEATURE_DIMS, constants.QUERY_SIZE, bias=False)
        self.value_matrix_down = nn.Linear(constants.FEATURE_DIMS, constants.QUERY_SIZE, bias=False)
        self.dropout = nn.Dropout(constants.DROPOUT)

    def forward(self, feature_vectors):
        shared.check_eq(feature_vectors.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        if constants.DEBUG: print("ATTENTION")
        query_vectors = self.query_matrix(feature_vectors)
        key_vectors = self.key_matrix(feature_vectors)
        attention_matrix = query_vectors @ key_vectors.transpose(-2, -1)
        if constants.DEBUG: print(query_vectors.shape, key_vectors.shape, attention_matrix.shape)
        attention_matrix /= math.sqrt(constants.QUERY_SIZE)
        if constants.DEBUG: print(attention_matrix)
        causal_mask = torch.triu(torch.ones_like(attention_matrix, dtype=torch.bool), diagonal=1)
        attention_matrix = attention_matrix.masked_fill(causal_mask, -math.inf)
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        attention_matrix = self.dropout(attention_matrix)
        value_vectors = self.value_matrix_down(feature_vectors)
        enrichment_stack = attention_matrix @ value_vectors
        if constants.DEBUG: print(feature_vectors.shape, enrichment_stack.shape)
        return enrichment_stack
