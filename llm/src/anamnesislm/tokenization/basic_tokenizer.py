from __future__ import annotations

from collections.abc import Sequence

import tiktoken


class BasicTokenizer:
    """Thin wrapper around GPT-2 BPE tokenization via tiktoken."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding_name = encoding_name
        self._encoder = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode text without adding or accepting special tokens."""
        return self._encoder.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._encoder.decode(list(ids))

    @property
    def vocab_size(self) -> int:
        return self._encoder.n_vocab

    @property
    def eos_token_id(self) -> int:
        return self._encoder.eot_token
