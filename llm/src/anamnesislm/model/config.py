from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Transformer size. Start tiny to check data and loss, then scale (plan: ~30-60M params)."""

    vocab_size: int = 50257  # GPT-2 tokenizer
    context_length: int = 512  # must cover PipelineConfig.max_sequence_length
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    drop_rate: float = 0.1
    qkv_bias: bool = False
