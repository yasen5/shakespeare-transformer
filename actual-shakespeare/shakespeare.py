#!/usr/bin/env python
# coding: utf-8

text = open('input.txt', 'r').read()
text[:10]


# TODO try words instead of chars
unique_chars = sorted(list(set(text)))


tokenizer_map = {char : i for i,char in enumerate(unique_chars)}
untokenizer_map = {i : char for i,char in enumerate(unique_chars)}
encode = lambda string : [tokenizer_map[char] for char in string]
decode = lambda token_list: ''.join(
    untokenizer_map[int(token)] for token in token_list
)


import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(42)
full_text = torch.tensor(encode(text), dtype=torch.int64, device=device) # must be int64


# TODO tune
CONTEXT_WINDOW_SIZE = 32
N_BATCHES = 3
LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY = 0.999
EPOCHS = int(5e3)
N_ATTENTION_HEADS = 4
N_FEATURE_DIMS = 64
QUERY_SIZE = 4
N_UNIQUE_CHARS = len(unique_chars)
SEED = 42
N_HIDDEN_NEURONS = 200
DROPOUT = 0.01
DEBUG = False
# CONTEXT_WINDOW_SIZE = 3
# N_BATCHES = 3
# LEARNING_RATE = 1e-3
# LEARNING_RATE_DECAY = 0.999
# EPOCHS = int(1e0)
# N_ATTENTION_HEADS = 1
# N_FEATURE_DIMS = 4
# QUERY_SIZE = 4
# N_UNIQUE_CHARS = 65
# SEED = 42
# N_HIDDEN_NEURONS = 200
# DROPOUT = 0.01
# DEBUG = True


dev_cutoff = int(0.9 * len(full_text)) # TODO add test split
train_data = full_text[:dev_cutoff]
dev_data = full_text[dev_cutoff:]


def GetRandomBatch(data):
    batch_start_indices = torch.randint(
        low=0,
        high=len(data) - CONTEXT_WINDOW_SIZE,
        size=(N_BATCHES,),
        device=device,
    )
    offsets = torch.arange(CONTEXT_WINDOW_SIZE, device=device)
    batch_positions = batch_start_indices[:, None] + offsets
    inputs = data[batch_positions]
    labels = data[batch_positions + 1]
    if DEBUG: print("BATCH", inputs.shape, labels.shape)
    return inputs, labels


def LayerNorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-5)


class CheckError(RuntimeError):
    pass


def _normalize_for_check(x):
    # Treat torch.Size, tuple, and list shapes as equivalent.
    if isinstance(x, (list, tuple)):
        return tuple(x)

    # Avoid requiring torch import just for this helper.
    if type(x).__name__ == "Size" and type(x).__module__.startswith("torch"):
        return tuple(x)

    return x


def check_eq(left, right, left_name="left", right_name="right"):
    left_cmp = _normalize_for_check(left)
    right_cmp = _normalize_for_check(right)

    if left_cmp != right_cmp:
        raise CheckError(
            f"CHECK_EQ failed: {left_name} == {right_name}\n"
            f"  {left_name}:  {left!r}\n"
            f"  {right_name}: {right!r}"
        )


import math
import torch.nn as nn

class AttentionHead():
    def __init__(self):
        self.query_matrix = torch.randn((N_FEATURE_DIMS, QUERY_SIZE), device=device)
        self.key_matrix = torch.randn(self.query_matrix.shape, device=device)
        self.value_matrix_up = torch.randn((N_FEATURE_DIMS, QUERY_SIZE), device=device)
        self.value_matrix_down = torch.randn((QUERY_SIZE, N_FEATURE_DIMS), device=device)
        self.params = [self.query_matrix, self.key_matrix, self.value_matrix_up, self.value_matrix_down]
        for param in self.params:
            param.requires_grad = True
        # self.attention_matrix = torch.empty((CONTEXT_WINDOW_SIZE, CONTEXT_WINDOW_SIZE))

    def attend(self, feature_vectors):
        check_eq(feature_vectors.shape, [CONTEXT_WINDOW_SIZE, N_FEATURE_DIMS])
        if DEBUG: print("ATTENTION")
        query_vectors = feature_vectors @ self.query_matrix  # TODO pytorch vectorize
        key_vectors = feature_vectors @ self.key_matrix # TODO pytorch vectorize
        attention_matrix = query_vectors @ key_vectors.t()
        if DEBUG: print(query_vectors.shape, key_vectors.shape, attention_matrix.shape)

        attention_matrix /= math.sqrt(QUERY_SIZE)
        if DEBUG: print(attention_matrix)
        for row in range(attention_matrix.shape[0]):
            for col in range(attention_matrix.shape[1]):
                if (col > row):
                    attention_matrix[row,col] = -math.inf;
        attention_matrix = torch.softmax(attention_matrix, dim=1)
        enrichment_stack = torch.zeros(feature_vectors.shape, device=device)
        for row in range(attention_matrix.shape[0]):
            for col in range(attention_matrix.shape[1]):
                enrichment_stack[row] += attention_matrix[row,col] * self.value_matrix_up @ self.value_matrix_down @ feature_vectors[col]
        if DEBUG: print(feature_vectors.shape, enrichment_stack.shape)
        return (feature_vectors + enrichment_stack).view(-1, CONTEXT_WINDOW_SIZE, N_FEATURE_DIMS)


def XavierFactor(param):
    return math.sqrt(2 / (param.shape[0] + param.shape[1]))

class FeedForward():
    def __init__(self):
        self.W1 = torch.randn((N_ATTENTION_HEADS * N_FEATURE_DIMS, N_HIDDEN_NEURONS), device=device)
        self.W1 *= XavierFactor(self.W1)
        self.b1 = torch.zeros((N_HIDDEN_NEURONS,), device=device)
        self.W2 = torch.randn((N_HIDDEN_NEURONS, N_UNIQUE_CHARS), device=device)
        self.W2 *= XavierFactor(self.W2)
        self.b2 = torch.zeros((N_UNIQUE_CHARS,), device=device)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, context):
        # context: [BATCH_SIZE, CONTEXT_WINDOW_SIZE, N_FEATURE_DIMS]
        check_eq(context.shape, [N_BATCHES, CONTEXT_WINDOW_SIZE, N_ATTENTION_HEADS * N_FEATURE_DIMS])
        if DEBUG: print("FORWARD")
        if DEBUG: print(context.shape)
        hidden_output = context @ self.W1 + self.b1
        if DEBUG: print(hidden_output.shape)
        hidden_output = torch.where(hidden_output > 0, hidden_output, 0)
        if DEBUG: print(hidden_output.shape)
        output = hidden_output @ self.W2 + self.b2
        if DEBUG: print(output.shape)
        check_eq(output.shape, [N_BATCHES, CONTEXT_WINDOW_SIZE, N_UNIQUE_CHARS])
        return output


import torch.nn.functional as F

def ResetGrad(params):
    for param in params:
        param.grad = None

def ApplyGrad(params, learning_rate):
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad

class Transformer():
    def __init__(self):
        self.feature_embedding_table = torch.randn((N_UNIQUE_CHARS, N_FEATURE_DIMS), device=device)
        self.feature_embedding_table *= XavierFactor(self.feature_embedding_table)
        self.feature_mixer = torch.randn((N_ATTENTION_HEADS * N_FEATURE_DIMS, N_ATTENTION_HEADS * N_FEATURE_DIMS), device=device)
        self.attention_heads = [AttentionHead() for _ in range(N_ATTENTION_HEADS)]
        self.feed_forward = FeedForward()
        self.params = [self.feature_embedding_table, self.feature_mixer] + self.feed_forward.params 
        for attention_head in self.attention_heads:
            self.params += attention_head.params
        for param in self.params:
            param.requires_grad = True
        self.learning_rate = LEARNING_RATE
        self.optimizer = torch.optim.Adam(self.params, lr=self.learning_rate)
        self.dropout = nn.Dropout(DROPOUT)
        # self.attended_feature_vectors = torch.empty((N_BATCHES, CONTEXT_WINDOW_SIZE, N_ATTENTION_HEADS * N_FEATURE_DIMS))

    def forward(self, context):
        feature_vectors = self.feature_embedding_table[context]
        attended_feature_vectors = torch.empty(
            (N_BATCHES, CONTEXT_WINDOW_SIZE, N_ATTENTION_HEADS * N_FEATURE_DIMS),
            device=device,
        )
        for i in range(N_BATCHES):
            attended_feature_vectors[i] = LayerNorm(torch.concat([attention_head.attend(feature_vectors[i]) for attention_head in self.attention_heads], dim=-1));
        if DEBUG: print("Attended feature vectors: ", attended_feature_vectors.shape)
        mixed_feature_vectors = attended_feature_vectors @ self.feature_mixer
        if DEBUG: print("Mixed: ", mixed_feature_vectors.shape)
        mixed_feature_vectors = self.dropout(mixed_feature_vectors)
        if DEBUG: print("Dropout: ", mixed_feature_vectors.shape)
        output = self.feed_forward.forward(LayerNorm(mixed_feature_vectors))
        if DEBUG: print("FF: ", output.shape)
        return output

    def backward(self, output, label):
        loss = F.cross_entropy(output, label)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def summarize(self):
        print("=====ATTENTION=====")
        print(f"For each {N_ATTENTION_HEADS} head, query matrix is {self.attention_heads[0].query_matrix.shape}, key matrix is {self.attention_heads[0].key_matrix.shape}, value matrix is {self.attention_heads[0].value_matrix_up.shape} x {self.attention_heads[0].value_matrix_down.shape}")


transformer = Transformer()
X_batch, Y_batch = GetRandomBatch(train_data)


transformer.summarize()


for i in range(EPOCHS):
    X_batch, Y_batch = GetRandomBatch(train_data);
    out = transformer.forward(X_batch)
    B, T, C = out.shape
    loss = transformer.backward(out.view(B*T, C), Y_batch.view(B*T))
    if (i % 1 == 0):
        print(f"loss: {loss}")


def DecodeTokenList(token_list):
    return ''.join(untokenizer_map[int(token)] for token in token_list)


def TestModel(transformer, data, n_tokens=10):
    X_test, _ = GetRandomBatch(data)
    # X_test = X_batch
    context = X_test[0].clone()
    start_context = context.clone()

    print("START:")
    print(DecodeTokenList(context))

    generated_tokens = []

    for i in range(n_tokens):
        # Add fake batch dimension: [T] -> [1, T]
        context_batch = context.view(1, CONTEXT_WINDOW_SIZE)

        # Your transformer currently expects N_BATCHES exactly,
        # so repeat the same context N_BATCHES times.
        context_batch = context_batch.repeat(N_BATCHES, 1)

        # Predict next token from the final position in the first batch row
        with torch.no_grad():
            logits = transformer.forward(context_batch)
        pred = logits[0, -1].argmax(dim=-1)

        generated_tokens.append(pred)

        # Slide context window left and append prediction
        context = torch.cat([context[1:], pred.view(1)], dim=0)

        print(f"PRED {i + 1}: {untokenizer_map[int(pred)]!r}")

    print("\nGENERATED:")
    print(DecodeTokenList(generated_tokens))

    print("\nFULL:")
    print(DecodeTokenList(start_context) + DecodeTokenList(generated_tokens))

TestModel(transformer, train_data, n_tokens=10)


