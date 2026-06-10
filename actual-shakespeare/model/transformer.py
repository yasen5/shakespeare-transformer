import torch.nn.functional as F
import torch
import constants
import shared
import attention
import feedforward
import math

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
        self.feature_embedding_table = torch.randn((constants.N_UNIQUE_CHARS, constants.N_FEATURE_DIMS), device=shared.device)
        self.feature_embedding_table *= shared.XavierFactor(self.feature_embedding_table)
        self.feature_mixer = torch.randn((constants.N_ATTENTION_HEADS * constants.N_FEATURE_DIMS, constants.N_ATTENTION_HEADS * constants.N_FEATURE_DIMS), device=shared.device)
        self.feature_mixer *= shared.XavierFactor(self.feature_embedding_table)
        self.attention_heads = [attention.AttentionHead() for _ in range(constants.N_ATTENTION_HEADS)]
        self.feed_forward = feedforward.FeedForward()
        self.params = [self.feature_embedding_table, self.feature_mixer] + self.feed_forward.params 
        for attention_head in self.attention_heads:
            self.params += attention_head.params
        for param in self.params:
            param.requires_grad = True
        self.positional_encoding = torch.empty((constants.CONTEXT_WINDOW_SIZE, constants.N_FEATURE_DIMS), device=shared.device)
        with torch.no_grad():
            for position in range(constants.CONTEXT_WINDOW_SIZE):
                self.positional_encoding[position] = torch.tensor([
                    math.sin(position / math.pow(10000, 2 * (i // 2) / constants.N_FEATURE_DIMS))
                    if i % 2 == 0
                    else math.cos(position / math.pow(10000, 2 * (i // 2) / constants.N_FEATURE_DIMS))
                    for i in range(constants.N_FEATURE_DIMS)
                ], device=shared.device)
        self.positional_encoding.requires_grad_(False)
        self.learning_rate = constants.LEARNING_RATE
        self.optimizer = torch.optim.Adam(self.params, lr=self.learning_rate)
        self.dropout = torch.nn.Dropout(constants.DROPOUT)
        # self.attended_feature_vectors = torch.empty((N_BATCHES, CONTEXT_WINDOW_SIZE, constants.N_ATTENTION_HEADS * constants.N_FEATURE_DIMS))

    def forward(self, context):
        feature_vectors = self.feature_embedding_table[context]
        feature_vectors = feature_vectors + self.positional_encoding
        attended_feature_vectors = torch.empty((constants.N_BATCHES, constants.CONTEXT_WINDOW_SIZE, constants.N_ATTENTION_HEADS * constants.N_FEATURE_DIMS), device=shared.device)
        for i in range(constants.N_BATCHES):
            attended_feature_vectors[i] = shared.LayerNorm(torch.concat([attention_head.attend(feature_vectors[i]) for attention_head in self.attention_heads], dim=-1));
        if constants.DEBUG: print("Attended feature vectors: ", attended_feature_vectors.shape)
        mixed_feature_vectors = attended_feature_vectors @ self.feature_mixer
        if constants.DEBUG: print("Mixed: ", mixed_feature_vectors.shape)
        mixed_feature_vectors = self.dropout(mixed_feature_vectors)
        if constants.DEBUG: print("Dropout: ", mixed_feature_vectors.shape)
        output = self.feed_forward.forward(shared.LayerNorm(mixed_feature_vectors))
        if constants.DEBUG: print("FF: ", output.shape)
        return output
    
    def backward(self, output, label):
        loss = F.cross_entropy(output, label)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def summarize(self):
        print("=====ATTENTION=====")
        print(f"For each {constants.N_ATTENTION_HEADS} head, query matrix is {self.attention_heads[0].query_matrix.shape}, key matrix is {self.attention_heads[0].key_matrix.shape}, value matrix is {self.attention_heads[0].value_matrix_up.shape} x {self.attention_heads[0].value_matrix_down.shape}")