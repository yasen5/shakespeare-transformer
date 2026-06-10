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
    batch_start_indices = torch.randint(low=0, high=len(data) - constants.CONTEXT_WINDOW_SIZE, size=(constants.N_BATCHES,), device=shared.device)
    offsets = torch.arange(constants.CONTEXT_WINDOW_SIZE, device=shared.device)
    inputs = data[batch_start_indices[:, None] + offsets]
    labels = data[batch_start_indices[:, None] + offsets + 1]
    if constants.DEBUG: print("BATCH", inputs.shape, labels.shape)
    return inputs, labels
