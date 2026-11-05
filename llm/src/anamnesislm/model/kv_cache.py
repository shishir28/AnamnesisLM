"""Keys and values kept between generation steps (a "KV cache").

Without a cache, every new token re-runs the whole sequence through the model. With one, each
attention layer stores the keys and values it has already computed, so a step only processes
the newest token and attends to the stored ones. Same maths, far less repeated work.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class LayerCache:
    """One attention layer's keys and values so far, each (batch, heads, seq_len, head_dim)."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Number of positions stored."""
        return 0 if self.keys is None else self.keys.shape[2]


@dataclass
class KVCache:
    """One LayerCache per transformer block. Create a new one for every sequence."""

    num_layers: int
    layers: list[LayerCache] = field(init=False)

    def __post_init__(self) -> None:
        self.layers = [LayerCache() for _ in range(self.num_layers)]

    @property
    def length(self) -> int:
        """Number of positions already processed (the next token's position)."""
        return self.layers[0].length if self.layers else 0
