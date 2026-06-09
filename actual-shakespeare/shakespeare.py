#!/usr/bin/env python
# coding: utf-8

text = open('input.txt', 'r').read()
text[:10]


# TODO try words instead of chars
unique_chars = sorted(list(set(text)))


tokenizer_map = {char : i for i,char in enumerate(unique_chars)}
untokenizer_map = {i : char for i,char in enumerate(unique_chars)}
encode = lambda string : [tokenizer_map[char] for char in string]
decode = lambda token_list : [untokenizer_map[token] for token in token_list]


import torch
full_text = torch.tensor(encode(text), dtype=torch.int64) # must be int64


# TODO tune
CONTEXT_WINDOW_SIZE = 32
N_BATCHES = 3
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 0.999
EPOCHS = int(1e5)
N_ATTENTION_HEADS = 4
N_FEATURE_DIMS = 64
QUERY_SIZE = 4
N_UNIQUE_CHARS = 65
SEED = 42
N_HIDDEN_NEURONS = 200
DROPOUT = 0.01


dev_cutoff = int(0.9 * len(full_text)) # TODO add test split
train_data = full_text[:dev_cutoff]
dev_data = full_text[dev_cutoff:]


def GetRandomBatch(data):
    batch_start_indices =  torch.randint(low=0, high=len(data) - CONTEXT_WINDOW_SIZE, size=(N_BATCHES,))
    inputs = torch.stack([data[i:i+CONTEXT_WINDOW_SIZE] for i in batch_start_indices])
    # labels = torch.stack([data[i+1:i+CONTEXT_WINDOW_SIZE+1] for i in batch_start_indices])
    labels = torch.stack([data[i+CONTEXT_WINDOW_SIZE] for i in batch_start_indices])
    print(inputs.shape, labels.shape)
    return inputs, labels


def LayerNorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-5)


import math
import copy
import torch.nn as nn

class AttentionHead():
    def __init__(self):
        self.query_matrix = torch.randn((N_FEATURE_DIMS, QUERY_SIZE))
        self.key_matrix = torch.randn(self.query_matrix.shape)
        self.value_matrix_up = torch.randn((N_FEATURE_DIMS, QUERY_SIZE))
        self.value_matrix_down = torch.randn((QUERY_SIZE, N_FEATURE_DIMS))
        self.params = [self.query_matrix, self.key_matrix, self.value_matrix_up, self.value_matrix_down]
        for param in self.params:
            param.requires_grad = True

        self.query_vectors = torch.empty((QUERY_SIZE, CONTEXT_WINDOW_SIZE))
        self.key_vectors = torch.empty((QUERY_SIZE, CONTEXT_WINDOW_SIZE))
        self.attention_matrix = torch.empty((CONTEXT_WINDOW_SIZE, CONTEXT_WINDOW_SIZE))
        self.attended_feature_vectors = torch.empty((CONTEXT_WINDOW_SIZE, N_FEATURE_DIMS))

    def attend(self, feature_vectors):
        print("ATTENTION")
        self.attended_feature_vectors = copy.deepcopy(feature_vectors)
        query_vectors = feature_vectors @ self.query_matrix  # TODO pytorch vectorize
        key_vectors = feature_vectors @ self.key_matrix # TODO pytorch vectorize
        attention_matrix = query_vectors @ key_vectors.t()
        print(query_vectors.shape, key_vectors.shape, attention_matrix.shape)

        attention_matrix /= math.sqrt(N_FEATURE_DIMS)
        for row in range(attention_matrix.shape[0]):
            for col in range(attention_matrix.shape[1]):
                if (col > row):
                    attention_matrix[row,col] = -math.inf;
        attention_matrix = torch.softmax(attention_matrix, dim=0)
        for row in range(CONTEXT_WINDOW_SIZE):
            for col in range(CONTEXT_WINDOW_SIZE):
                self.attended_feature_vectors[row] += attention_matrix[row,col] * self.value_matrix_up @ self.value_matrix_down @ feature_vectors[row]
        return self.attended_feature_vectors.view(-1, CONTEXT_WINDOW_SIZE, N_FEATURE_DIMS)


class FeedForward():
    def __init__(self):
        g = torch.Generator().manual_seed(SEED) # for reproducibility
        self.W1 = torch.randn((N_ATTENTION_HEADS * N_FEATURE_DIMS, N_HIDDEN_NEURONS), generator=g)
        self.b1 = torch.randn((N_HIDDEN_NEURONS), generator=g)
        self.W2 = torch.randn((N_HIDDEN_NEURONS, N_UNIQUE_CHARS), generator=g)
        self.b2 = torch.randn((N_UNIQUE_CHARS,), generator=g)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, context):
        # context: [BATCH_SIZE, CONTEXT_WINDOW_SIZE, N_FEATURE_DIMS]
        print("FORWARD")
        print(context.shape)
        hidden_output = context @ self.W1 + self.b1
        print(hidden_output.shape)
        hidden_output = torch.where(hidden_output > 0, hidden_output, 0)
        print(hidden_output.shape)
        output = hidden_output @ self.W2 + self.b2
        print(output.shape)
        hidden_output = self.dropout(hidden_output)
        return output


import torch.nn.functional as F

def ResetGrad(params):
    for param in params:
        param.grad = None

def ApplyGrad(params, learning_rate):
    for param in params:
        param.data -= learning_rate * param.grad

class Transformer():
    def __init__(self):
        g = torch.Generator().manual_seed(SEED) # for reproducibility
        self.feature_embedding_table = torch.randn((N_UNIQUE_CHARS, N_FEATURE_DIMS))
        self.feature_mixer = torch.randn((N_ATTENTION_HEADS * N_FEATURE_DIMS, N_ATTENTION_HEADS * N_FEATURE_DIMS))
        self.attention_heads = [AttentionHead() for _ in range(N_ATTENTION_HEADS)]
        self.feed_forward = FeedForward()
        self.params = [self.feature_embedding_table, self.feature_mixer] + self.feed_forward.params 
        for attention_head in self.attention_heads:
            self.params += attention_head.params
        for param in self.params:
            param.requires_grad = True
        self.learning_rate = LEARNING_RATE
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, context):
        feature_vectors = self.feature_embedding_table[context]
        attended_feature_vectors = LayerNorm(torch.concat([attention_head.attend(feature_vectors.view(N_BATCHES * CONTEXT_WINDOW_SIZE, -1)) for attention_head in self.attention_heads], dim=-1))
        print("Attended feature vectors: ", attended_feature_vectors.shape)
        mixed_feature_vectors = attended_feature_vectors @ self.feature_mixer
        print("Mixed: ", mixed_feature_vectors.shape)
        mixed_feature_vectors = self.dropout(mixed_feature_vectors)
        print("Dropout: ", mixed_feature_vectors.shape)
        output = LayerNorm(self.feed_forward.forward(mixed_feature_vectors))
        print("FF: ", output.shape)
        return output[:, 0, :]

    def backward(self, output, label):
        loss = F.cross_entropy(output, label)
        ResetGrad(self.params)
        loss.backward()
        ApplyGrad(self.params, self.learning_rate)

    def summarize(self):
        print("=====ATTENTION=====")
        print(f"For each {N_ATTENTION_HEADS} head, query matrix is {self.attention_heads[0].query_matrix.shape}, key matrix is {self.attention_heads[0].key_matrix.shape}, value matrix is {self.attention_heads[0].value_matrix_up.shape} x {self.attention_heads[0].value_matrix_down.shape}")


transformer = Transformer()
X_batch, Y_batch = GetRandomBatch(train_data)


transformer.summarize()


transformer.backward(transformer.forward(X_batch), Y_batch)


X_batch.shape




