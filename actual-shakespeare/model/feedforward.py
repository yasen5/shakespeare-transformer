import torch
from . import constants
from . import shared
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(constants.SEED) # for reproducibility
        self.linear1 = nn.Linear(constants.FEATURE_DIMS, 4 * constants.FEATURE_DIMS)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * constants.FEATURE_DIMS, constants.FEATURE_DIMS)
        self.dropout = nn.Dropout(constants.DROPOUT)

    def forward(self, context):
        # context: [BATCH_SIZE, CONTEXT_WINDOW_SIZE, FEATURE_DIMS]
        shared.check_eq(context.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        if constants.DEBUG: print("FORWARD")
        output = self.linear1(context)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        shared.check_eq(output.shape, [constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS])
        return output
