import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.w1 = nn.Linear(784, 512)
        self.a1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.w2 = nn.Linear(512, 10)
        self.a2 = nn.Sigmoid()
        pass

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        out_w1 = self.w1(images)
        out_a1 = self.a1(out_w1)
        out_dropout1 = self.dropout1(out_a1)
        out_w2 = self.w2(out_dropout1)
        out_a2 = self.a2(out_w2)
        return torch.round(out_a2, decimals=4)
        pass
