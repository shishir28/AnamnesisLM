import torch

from anamnesislm.model.gpt import GPTModel
from anamnesislm.model.kv_cache import KVCache


def generate(
    model: GPTModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int,
) -> list[int]:
    """Greedy decoding: repeatedly append the most likely next token.

    Run under torch.no_grad() with model.eval(). Stops at eos_token_id or after max_new_tokens.
    Returns only the new tokens, without EOS.

    Multi-head models use a KV cache: the prompt is processed once, then each step feeds only
    the newest token. Single-head models (old checkpoints), and any step where the sequence has
    grown past context_length, re-run the last context_length tokens instead, as before.
    """
    device = next(model.parameters()).device
    tokens = list(prompt_ids)
    new_ids: list[int] = []
    cache = KVCache(model.config.num_layers) if model.config.multi_head else None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_token_id = _next_token(_last_logits(model, tokens, cache, device))
            if next_token_id == eos_token_id:
                break
            tokens.append(next_token_id)
            new_ids.append(next_token_id)
    return new_ids


def _last_logits(
    model: GPTModel, tokens: list[int], cache: KVCache | None, device: torch.device
) -> torch.Tensor:
    """(1, vocab_size) logits for the token after tokens."""
    if cache is not None and len(tokens) <= model.config.context_length:
        unseen = tokens[cache.length :]  # the whole prompt on the first step, then one token
        logits: torch.Tensor = model(torch.tensor([unseen], device=device), cache)
    else:
        window = tokens[-model.config.context_length :]
        logits = model(torch.tensor([window], device=device))
    return logits[:, -1]


def _next_token(logits: torch.Tensor) -> int:
    """Most likely next token for (1, vocab_size) logits."""
    return int(logits.argmax(dim=-1).item())
