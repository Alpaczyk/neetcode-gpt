import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        torch.no_grad()
        stats = []
        for layer in model:
            x = layer.forward(x)
            if isinstance(layer, nn.Linear):
                mean = torch.round(torch.mean(x), decimals=4).item()
                std = torch.round(torch.std(x), decimals=4).item()
                if x.dim() >= 2:
                    dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                else:
                    dead_frac = round((x <= 0).float().mean().item(), 4)
                res = {"mean": mean, "std": std, "dead_fraction": dead_frac}
                stats.append(res)
        return stats

        pass

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.zero_grad()
        outputs = model.forward(x)
        loss = nn.MSELoss()(outputs, y)
        loss.backward()
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean_val = round(grad.mean().item(), 4)
                std_val = round(grad.std().item(), 4)
                norm_val = round(torch.norm(grad).item(), 4)
                stats.append({'mean': mean_val, 'std': std_val, 'norm': norm_val})
        return stats
        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for act in activation_stats:
            if act['dead_fraction'] > 0.5:
                return 'dead_neurons'
            if act['std'] < 0.1:
                return 'vanishing_gradients'
            if act['std'] > 10.0:
                return 'exploding_gradients'
        for grad in gradient_stats:
            if grad['norm'] > 1000:
                return 'exploding_gradients'
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        return "healthy"
        pass
