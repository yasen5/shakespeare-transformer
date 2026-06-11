from . import feedforward
from . import attention
from . import constants
from . import shared
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_heads = nn.ModuleList([attention.AttentionHead() for _ in range(constants.N_ATTENTION_HEADS)])
        self.feedforward = feedforward.FeedForward()
        self.projection = nn.Linear(constants.FEATURE_DIMS, constants.FEATURE_DIMS)
        self.dropout = torch.nn.Dropout(constants.DROPOUT)
        self.layer_norm1 = nn.LayerNorm(constants.FEATURE_DIMS)
        self.layer_norm2 = nn.LayerNorm(constants.FEATURE_DIMS)
    
    def forward(self, position_encoded_feature_vectors):
        shared.check_eq(position_encoded_feature_vectors.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        normed_position_encoded_feature_vectors = self.layer_norm1(position_encoded_feature_vectors)
        if constants.DEBUG: print(normed_position_encoded_feature_vectors.shape)
        attended_feature_vectors = position_encoded_feature_vectors + self.dropout(torch.concat([attention_head(normed_position_encoded_feature_vectors) for attention_head in self.attention_heads], dim=-1))
        if constants.DEBUG: print(attended_feature_vectors.shape)
        out = position_encoded_feature_vectors + self.feedforward.forward(self.layer_norm2(attended_feature_vectors))
        return out
        
