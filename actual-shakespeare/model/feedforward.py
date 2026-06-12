import torch
from . import constants
from . import shared
import torch.nn as nn

class FeedForward():
    def __init__(self):
        torch.manual_seed(constants.SEED) # for reproducibility
        self.W1 = torch.randn((constants.N_ATTENTION_HEADS * constants.FEATURE_DIMS, constants.FEATURE_DIMS), device=shared.device)
        self.W1 *= shared.XavierFactor(self.W1)
        self.b1 = torch.zeros((constants.FEATURE_DIMS,), device=shared.device)
        self.W2 = torch.randn((constants.FEATURE_DIMS, constants.FEATURE_DIMS), device=shared.device)
        self.W2 *= shared.XavierFactor(self.W2)
        self.b2 = torch.zeros((constants.FEATURE_DIMS,), device=shared.device)
        self.relu = nn.ReLU()
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, context):
        shared.check_eq(context.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        if constants.DEBUG: print("FORWARD")
        if constants.DEBUG: print(context.shape)
        out = context @ self.W1 + self.b1
        if constants.DEBUG: print(out.shape)
        out = self.relu(out)
        out = out @ self.W2 + self.b2
        out = nn.functional.dropout(out)
        if constants.DEBUG: print(out.shape)
        shared.check_eq(out.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        return out
