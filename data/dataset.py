import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        vocab = raw_dataset.split()
        indices = torch.randint(len(vocab) - context_length, (batch_size,))

        offsets = torch.arange(context_length)
        x_indices = indices.unsqueeze(1) + offsets
        
        X = [[vocab[x] for x in x_indic] for x_indic in x_indices]
        Y =  [[vocab[x] for x in x_indic] for x_indic in x_indices + 1]
        return X,Y
        pass
