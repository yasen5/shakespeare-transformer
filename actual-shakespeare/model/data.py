import torch
from . import constants
from . import shared
text = open('input.txt', 'r').read()
text[:10]
unique_chars = sorted(list(set(text)))
tokenizer_map = {char : i for i,char in enumerate(unique_chars)}
untokenizer_map = {i : char for i,char in enumerate(unique_chars)}
encode = lambda string : [tokenizer_map[char] for char in string]
decode = lambda token_list: ''.join(
    untokenizer_map[int(token)] for token in token_list
)

full_text = torch.tensor(encode(text), dtype=torch.int64, device=shared.device) # must be int64



def GetRandomBatch(data):
    ix = torch.randint(len(data) - constants.CONTEXT_WINDOW_SIZE, (constants.BATCH_SIZE,))
    x = torch.stack([data[i:i+constants.CONTEXT_WINDOW_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+constants.CONTEXT_WINDOW_SIZE+1] for i in ix])
    x, y = x.to(shared.device), y.to(shared.device)
    return x, y
