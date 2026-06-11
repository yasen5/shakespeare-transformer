import torch.nn.functional as F
import torch
from . import constants
from . import shared
from . import attention
from . import feedforward
from . import transformer_block
import torch.nn as nn

def ResetGrad(params):
    for param in params:
        param.grad = None

def ApplyGrad(params, learning_rate):
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(constants.SEED) # for reproducibility
        self.feature_embedding_table = nn.Embedding(constants.N_UNIQUE_CHARS, constants.FEATURE_DIMS)
        self.positional_embedding_table = nn.Embedding(constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS)
        self.final_dense = nn.Linear(constants.FEATURE_DIMS, constants.N_UNIQUE_CHARS)
        self.transformer_blocks = nn.Sequential(*[transformer_block.TransformerBlock() for _ in range(constants.N_TRANSFORMER_BLOCKS)])
        self.learning_rate = constants.LEARNING_RATE
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context):
        if constants.DEBUG: print(context.shape)
        feature_vectors = self.feature_embedding_table(context)
        if constants.DEBUG: print(feature_vectors.shape)
        positional_embedding = self.positional_embedding_table(torch.arange(constants.CONTEXT_WINDOW_SIZE, device=shared.device))
        positionally_encoded_feature_vectors = feature_vectors + positional_embedding
        output = positionally_encoded_feature_vectors
        for block in self.transformer_blocks:
            output = block.forward(output)
            if constants.DEBUG : print("Finished block, output shape: ", output.shape)
        return self.final_dense(output)
    
    def backward(self, logits, label):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = label.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return loss
