import torch
from torch import nn

from anamnesislm.model.config import ModelConfig
from anamnesislm.model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Decoder-only transformer, randomly initialised.

    token + position embeddings -> num_layers x TransformerBlock -> LayerNorm -> vocab logits.
    Batches are right-padded, so the causal mask already stops real tokens attending to padding.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.drop_rate)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.out_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.out_head.weight = self.token_embedding.weight  # weight tying

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            token_ids: (batch_size, seq_length) token ids; seq_length <= context_length.

        Returns:
            (batch_size, seq_length, vocab_size) next-token logits for each position.
        """
        _, seq_length = token_ids.shape

        if seq_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds model context length {self.config.context_length}"
            )

        # (batch_size, seq_length, embedding_dim)
        token_embeddings = self.token_embedding(token_ids)
        # (seq_length,); broadcasts over the batch when added
        position_ids = torch.arange(seq_length, device=token_ids.device)
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_layer_norm(x)
        logits: torch.Tensor = self.out_head(x)

        return logits

    def num_parameters(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
