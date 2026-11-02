"""K independent MLPs stacked into one module; every layer is a single batched matmul over members.

Weights are stored (K, fan_in, fan_out) and biases (K, 1, fan_out). Initialization is the PyTorch default for
nn.Linear, U(-1/sqrt(fan_in), +1/sqrt(fan_in)) for weights and biases, drawn on CPU per member from its own seed
(members with the same seed get identical values), so the init is identical on any device.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedMLP(nn.Module):
    def __init__(self, num_members: int, in_dim: int, depth: int, width: int, leaky_slope: float):
        super().__init__()
        if depth < 1:
            raise ValueError(f"BatchedMLP: depth must be >= 1, got {depth}")
        self.num_members, self.in_dim, self.depth, self.width = num_members, in_dim, depth, width
        self.leaky_slope = leaky_slope
        dims = [in_dim] + [width] * depth + [1]
        self.layer_dims = list(zip(dims[:-1], dims[1:]))
        self.weights = nn.ParameterList(
            nn.Parameter(torch.zeros(num_members, fi, fo)) for fi, fo in self.layer_dims)
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(num_members, 1, fo)) for _, fo in self.layer_dims)

    def ordered_parameters(self) -> list[tuple[nn.Parameter, bool]]:
        """(parameter, is_weight_matrix) in a fixed order; weight decay applies to weight matrices only."""
        out = []
        for w, b in zip(self.weights, self.biases):
            out += [(w, True), (b, False)]
        return out

    @torch.no_grad()
    def initialize(self, member_seeds: list[int]) -> None:
        if len(member_seeds) != self.num_members:
            raise ValueError(f"initialize: {len(member_seeds)} seeds for {self.num_members} members")
        cache: dict[int, list[torch.Tensor]] = {}
        for k, seed in enumerate(member_seeds):
            if seed not in cache:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                tensors = []
                for fi, fo in self.layer_dims:
                    bound = 1.0 / math.sqrt(fi)
                    tensors.append(torch.empty(fi, fo).uniform_(-bound, bound, generator=gen))
                    tensors.append(torch.empty(1, fo).uniform_(-bound, bound, generator=gen))
                cache[seed] = tensors
            for (p, _), value in zip(self.ordered_parameters(), cache[seed]):
                p[k].copy_(value)

    @property
    def params_per_member(self) -> int:
        return sum(fi * fo + fo for fi, fo in self.layer_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (K, B, in_dim) -> (K, B)."""
        h = x
        last = len(self.layer_dims) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b, h, w)
            if i < last:
                h = F.leaky_relu(h, self.leaky_slope)
        return h.squeeze(-1)
