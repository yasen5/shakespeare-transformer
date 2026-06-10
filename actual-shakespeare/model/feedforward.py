import torch
from . import constants
from . import shared
import torch.nn as nn

class FeedForward():
    def __init__(self):
        torch.manual_seed(constants.SEED) # for reproducibility
        self.W1 = torch.randn((constants.N_ATTENTION_HEADS * constants.N_FEATURE_DIMS, constants.N_HIDDEN_NEURONS), device=shared.device)
        self.W1 *= shared.XavierFactor(self.W1)
        self.b1 = torch.zeros((constants.N_HIDDEN_NEURONS,), device=shared.device)
        self.W2 = torch.randn((constants.N_HIDDEN_NEURONS, constants.N_UNIQUE_CHARS), device=shared.device)
        self.W2 *= shared.XavierFactor(self.W2)
        self.b2 = torch.zeros((constants.N_UNIQUE_CHARS,), device=shared.device)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.dropout = nn.Dropout(constants.DROPOUT)

    def forward(self, context):
        # context: [BATCH_SIZE, CONTEXT_WINDOW_SIZE, N_FEATURE_DIMS]
        shared.check_eq(context.shape, [constants.N_BATCHES, constants.CONTEXT_WINDOW_SIZE, constants.N_ATTENTION_HEADS * constants.N_FEATURE_DIMS])
        if constants.DEBUG: print("FORWARD")
        if constants.DEBUG: print(context.shape)
        hidden_output = context @ self.W1 + self.b1
        if constants.DEBUG: print(hidden_output.shape)
        hidden_output = torch.where(hidden_output > 0, hidden_output, 0)
        if constants.DEBUG: print(hidden_output.shape)
        output = hidden_output @ self.W2 + self.b2
        if constants.DEBUG: print(output.shape)
        shared.check_eq(output.shape, [constants.N_BATCHES, constants.CONTEXT_WINDOW_SIZE, constants.N_UNIQUE_CHARS])
        return output
