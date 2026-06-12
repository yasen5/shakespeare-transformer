import torch.nn.functional as F
import torch
from . import constants
from . import shared
from . import attention
from . import feedforward
from . import transformer_block

def ResetGrad(params):
    for param in params:
        param.grad = None

def ApplyGrad(params, learning_rate):
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad

class Transformer():
    def __init__(self):
        torch.manual_seed(constants.SEED) # for reproducibility
        self.feature_embedding_table = torch.randn((constants.N_UNIQUE_CHARS, constants.FEATURE_DIMS), device=shared.device)
        self.feature_embedding_table *= shared.XavierFactor(self.feature_embedding_table)
        self.final_dense = torch.randn((constants.FEATURE_DIMS, constants.N_UNIQUE_CHARS), device=shared.device)
        self.final_dense *= shared.XavierFactor(self.final_dense)
        self.transformer_blocks = [transformer_block.TransformerBlock() for _ in range(constants.N_TRANSFORMER_BLOCKS)]
        self.params = [self.feature_embedding_table, self.final_dense]
        for block in self.transformer_blocks:
            self.params += block.params
        for param in self.params:
            param.requires_grad = True
        # positions = torch.arange(constants.CONTEXT_WINDOW_SIZE, device=shared.device).unsqueeze(1)
        # feature_indices = torch.arange(constants.FEATURE_DIMS, device=shared.device)
        # angles = positions / torch.pow(10000, 2 * (feature_indices // 2) / constants.FEATURE_DIMS)
        # self.positional_encoding = torch.where(feature_indices % 2 == 0, torch.sin(angles), torch.cos(angles))
        # self.positional_encoding.requires_grad_(False)
        self.positional_encoding = torch.randn((constants.CONTEXT_WINDOW_SIZE, constants.FEATURE_DIMS))
        self.learning_rate = constants.LEARNING_RATE
        self.optimizer = torch.optim.AdamW(self.params, lr=self.learning_rate)

    def forward(self, context):
        feature_vectors = self.feature_embedding_table[context]
        positionally_encoded_feature_vectors = feature_vectors + self.positional_encoding
        output = positionally_encoded_feature_vectors
        for block in self.transformer_blocks:
            output = block.forward(output)
            if constants.DEBUG : print("Finished block, output shape: ", output.shape)
        return output @ self.final_dense
    
    def backward(self, logits, label):
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), label.view(B*T))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()