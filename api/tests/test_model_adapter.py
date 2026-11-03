from types import SimpleNamespace
from unittest.mock import patch

import pytest

from anamnesis_api.application.extraction import InvalidNoteError
from anamnesis_api.infrastructure.model import LocalModelBackend


def make_backend(prompt_length=20):
    tokenizer = SimpleNamespace(encode=lambda text: [1] * prompt_length)
    model = SimpleNamespace(config=SimpleNamespace(context_length=512))
    extractor = SimpleNamespace(
        tokenizer=tokenizer, model=model, model_version="test", max_new_tokens=400
    )
    return LocalModelBackend(extractor, "cpu")


def test_context_budget_never_drops_note():
    backend = make_backend(prompt_length=200)
    with patch("anamnesislm.inference.Extractor") as extractor_cls:
        extractor_cls.return_value.extract.return_value = SimpleNamespace(
            raw_output='{"facts":[]}', facts=[], model_version="test", error=None
        )
        result = backend.extract("synthetic note", 400)
    assert extractor_cls.call_args.kwargs["max_new_tokens"] == 312
    assert result.generation_token_limit == 312
    assert result.warnings
    assert backend.extractor.max_new_tokens == 400  # Shared settings are unchanged.


def test_oversized_prompt_rejected_before_generation():
    backend = make_backend(prompt_length=512)
    with patch("anamnesislm.inference.Extractor") as extractor_cls:
        with pytest.raises(InvalidNoteError, match="too long"):
            backend.extract("synthetic note", 400)
        extractor_cls.assert_not_called()


def test_special_token_rejected_without_echoing_note():
    backend = make_backend()

    def reject(text):
        raise ValueError(text)

    backend.extractor.tokenizer.encode = reject
    with pytest.raises(InvalidNoteError) as exc:
        backend.extract("private marker <|endoftext|>", 400)
    assert "private marker" not in str(exc.value)
