import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Wk = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wq = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)
        pass

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.Wk(embedded)
        Q = self.Wq(embedded)
        V = self.Wv(embedded)
        _, seq_len, attn_dim = K.shape
        attn_score = (Q @ K.mT) / math.sqrt(attn_dim)
        caus_mask = torch.tril(torch.ones((seq_len, seq_len)))
        mask = caus_mask == 0
        attn_score = attn_score.masked_fill(mask, float('-inf'))
        out = self.softmax(attn_score)
        out = out @ V
        return torch.round(out, decimals=4)
        pass
