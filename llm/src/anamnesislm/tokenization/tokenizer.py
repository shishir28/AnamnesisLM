from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Stable tokenizer contract used by model and data components."""

    def encode(self, text: str) -> list[int]:
        """Encode ordinary text without accepting special tokens or appending EOS."""
        ...

    def decode(self, ids: Sequence[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...
