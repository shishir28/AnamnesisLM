import pytest
import tiktoken

from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer
from anamnesislm.tokenization.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> BasicTokenizer:
    return BasicTokenizer()


def test_uses_gpt2_encoding_by_default(tokenizer: BasicTokenizer) -> None:
    expected_encoding = tiktoken.get_encoding("gpt2")

    assert tokenizer.encoding_name == "gpt2"
    assert tokenizer.vocab_size == expected_encoding.n_vocab
    assert tokenizer.eos_token_id == expected_encoding.eot_token


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Patient denies chest pain.",
        "Dose: 81 mg—once daily.",
        "Patient says: café ☕",
    ],
)
def test_encode_decode_round_trip(tokenizer: BasicTokenizer, text: str) -> None:
    token_ids = tokenizer.encode(text)

    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert tokenizer.decode(token_ids) == text


def test_uses_requested_encoding() -> None:
    tokenizer = BasicTokenizer("cl100k_base")
    expected_encoding = tiktoken.get_encoding("cl100k_base")

    assert tokenizer.encoding_name == "cl100k_base"
    assert tokenizer.vocab_size == expected_encoding.n_vocab
    assert tokenizer.encode("Patient reports headache.") == expected_encoding.encode(
        "Patient reports headache."
    )


def test_rejects_invalid_encoding_name() -> None:
    with pytest.raises(ValueError):
        BasicTokenizer("not-a-real-encoding")


def test_decode_accepts_a_tuple(tokenizer: BasicTokenizer) -> None:
    token_ids = tokenizer.encode("Patient reports headache.")

    assert tokenizer.decode(tuple(token_ids)) == "Patient reports headache."


def test_rejects_end_of_text_in_input(tokenizer: BasicTokenizer) -> None:
    with pytest.raises(ValueError):
        tokenizer.encode("<|endoftext|>")


def test_encode_does_not_append_end_of_text(tokenizer: BasicTokenizer) -> None:
    token_ids = tokenizer.encode("Patient reports headache.")

    assert token_ids[-1] != tokenizer.eos_token_id


def test_matches_tiktoken_gpt2_encoding(tokenizer: BasicTokenizer) -> None:
    text = "Patient has a history of asthma."
    expected = tiktoken.get_encoding("gpt2").encode(text)

    assert tokenizer.encode(text) == expected


def test_implements_tokenizer_contract(tokenizer: BasicTokenizer) -> None:
    tokenizer_contract: Tokenizer = tokenizer

    assert isinstance(tokenizer_contract, Tokenizer)
