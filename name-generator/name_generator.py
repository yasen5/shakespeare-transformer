#!/usr/bin/env python
# coding: utf-8

words = open('names.txt', 'r').read().splitlines()
words[:10]


import torch
N_LETTERS: int = 27 # plus one for space


sorted_unique_characters = sorted(list(set(''.join(words))))
tokenizer_map = {char:index+1 for index,char in enumerate(sorted_unique_characters)}
tokenizer_map['.'] = 0
untokenizer_map = {index:char for char,index in tokenizer_map.items()}


CONTEXT_SIZE = 3;
BATCH_SIZE = 64;
FEATURE_DIMENSIONS = 10;
N_HIDDEN_WEIGHTS = 200;
STARTING_LEARNING_RATE = 0.1;
LEARNING_RATE_DECAY_RATE = 0.9999;
EPOCHS = int(10000)
LOG_RATE = 50


def build_dataset(words: list[str]):
    inputs, labels = [], []
    for word in words:
        chars = [0] * 3 + [tokenizer_map[char] for char in word] + [0]
        for i in range(0, len(chars) - CONTEXT_SIZE):
            inputs.append(chars[i:i+3])
            labels.append(chars[i+3])
    X = torch.tensor(inputs)
    Y = torch.tensor(labels)
    return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

X_train, Y_train = build_dataset(words[:n1])
X_val, Y_val = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])


for token_list in X_train[13:50]:
    print([untokenizer_map[int(token)] for token in token_list])


import torch.nn.functional as F

class PrimitiveModel():
    def __init__(self):
        g = torch.Generator().manual_seed(2147483647) # for reproducibility
        self.feature_lookup = torch.randn((N_LETTERS, FEATURE_DIMENSIONS), generator=g)
        self.hidden_weights = torch.randn((CONTEXT_SIZE * FEATURE_DIMENSIONS, N_HIDDEN_WEIGHTS), generator=g)
        self.hidden_biases = torch.randn((N_HIDDEN_WEIGHTS,), generator=g)
        self.output_weights = torch.randn((N_HIDDEN_WEIGHTS, N_LETTERS), generator=g)
        self.output_biases = torch.randn((N_LETTERS,), generator=g)
        self.parameters = [self.feature_lookup, self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases]
        for param in self.parameters:
            param.requires_grad = True
        self.learning_rate = STARTING_LEARNING_RATE
        self.backward_passes = 0
        self.loss_history = []
        self.x_axis = []

    def forward(self, context):
        hidden_output = torch.tanh(self.feature_lookup[context].view(-1, CONTEXT_SIZE * FEATURE_DIMENSIONS) @ self.hidden_weights + self.hidden_biases)
        output = hidden_output @ self.output_weights + self.output_biases
        return output
    
    def backward(self, output, label):
        loss = F.cross_entropy(output, label)
        for param in self.parameters:
            param.grad = None
        loss.backward()
        for param in self.parameters:
            param.data -= self.learning_rate * param.grad
        self.learning_rate *= LEARNING_RATE_DECAY_RATE
        if self.backward_passes % LOG_RATE == 0:
            self.loss_history.append(loss.log10().item())
            self.x_axis.append(self.backward_passes)
        self.backward_passes += 1

model = PrimitiveModel();


for i in range(0, 25000):
    batch_indices = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
    model.backward(model.forward(X_train[batch_indices]), Y_train[batch_indices])
    


sum(p.nelement() for p in model.parameters) # number of parameters in total


import matplotlib.pyplot as plt
plt.plot(model.x_axis, model.loss_history)


# sample from the model
g = torch.Generator().manual_seed(2147483647 + 9)

for _ in range(20):
    
    out = []
    context = [0] * CONTEXT_SIZE # initialize with all ...
    while True:
      logits = model.forward(context)
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(untokenizer_map[i] for i in out))




