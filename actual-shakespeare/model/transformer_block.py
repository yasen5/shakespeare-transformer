from . import feedforward
from . import attention
from . import constants
from . import shared
import torch

class TransformerBlock():
    def __init__(self):
        self.attention_heads = [attention.AttentionHead() for _ in range(constants.N_ATTENTION_HEADS)]
        self.feedforward = feedforward.FeedForward()
        self.layer_norm1 = torch.nn.LayerNorm(constants.FEATURE_DIMS, device=shared.device)
        self.layer_norm2 = torch.nn.LayerNorm(constants.FEATURE_DIMS, device=shared.device)
        self.params = self.feedforward.params + list(self.layer_norm1.parameters()) + list(self.layer_norm2.parameters())
        for attention_head in self.attention_heads:
            self.params += attention_head.params
    
    def forward(self, position_encoded_feature_vectors):
        shared.check_eq(position_encoded_feature_vectors.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        normed_position_encoded_feature_vectors = self.layer_norm1(position_encoded_feature_vectors)
        attended_feature_vectors = position_encoded_feature_vectors + torch.concat([attention_head.attend(normed_position_encoded_feature_vectors) for attention_head in self.attention_heads], dim=-1)
        return position_encoded_feature_vectors + self.feedforward.forward(self.layer_norm2(attended_feature_vectors))
        
