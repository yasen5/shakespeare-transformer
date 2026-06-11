from . import feedforward
from . import attention
from . import constants
from . import shared
import torch

class TransformerBlock():
    def __init__(self):
        self.attention_heads = [attention.AttentionHead() for _ in range(constants.N_ATTENTION_HEADS)]
        self.feedforward = feedforward.FeedForward()
        self.params = self.feedforward.params
        for attention_head in self.attention_heads:
            self.params += attention_head.params
    
    def forward(self, position_encoded_feature_vectors):
        shared.check_eq(position_encoded_feature_vectors.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.N_FEATURE_DIMS])
        normed_position_encoded_feature_vectors = shared.LayerNorm(position_encoded_feature_vectors)
        attended_feature_vectors = torch.concat([attention_head.attend(normed_position_encoded_feature_vectors) for attention_head in self.attention_heads], dim=-1)
        return self.feedforward.forward(shared.LayerNorm(attended_feature_vectors))
        
