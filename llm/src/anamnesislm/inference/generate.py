import torch

from anamnesislm.model.gpt import GPTModel


def generate(
    model: GPTModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int,
) -> list[int]:
    """Greedy decoding: repeatedly append the most likely next token.

    Run under torch.no_grad() with model.eval(). Each step feeds the last
    model.config.context_length tokens, takes logits[:, -1], picks argmax, and stops at
    eos_token_id or after max_new_tokens. Returns only the new tokens, without EOS.
    """
    device = next(model.parameters()).device
    tokens = list(prompt_ids)
    new_ids: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([tokens[-model.config.context_length :]], device=device)
            logits = model(input_ids)
            next_token_id = _next_token(logits[:, -1])
            if next_token_id == eos_token_id:
                break
            tokens.append(next_token_id)
            new_ids.append(next_token_id)
    return new_ids


def _next_token(logits: torch.Tensor) -> int:
    """Most likely next token for (1, vocab_size) logits."""
    return int(logits.argmax(dim=-1).item())
