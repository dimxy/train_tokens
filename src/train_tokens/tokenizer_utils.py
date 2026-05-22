from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import TokensConfig


def token_name(prefix: str, index: int) -> str:
    return f"{prefix}{index}|>"


def register_preference_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    labels: list[str],
    cfg: TokensConfig,
) -> tuple[dict[str, int], int]:
    """Add one special token per label; return (label→token_id, original_vocab_size).

    Labels are sorted before assignment so the mapping is deterministic regardless
    of the order they appear in the dataset.
    """
    original_vocab_size = len(tokenizer)
    sorted_labels = sorted(labels)
    new_tokens = [token_name(cfg.token_prefix, i) for i in range(len(sorted_labels))]

    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.resize_token_embeddings(len(tokenizer))
    _init_new_embeddings(model, original_vocab_size, cfg.init_strategy, tokenizer)

    label_to_token_id = {
        label: original_vocab_size + i for i, label in enumerate(sorted_labels)
    }
    return label_to_token_id, original_vocab_size


def _init_new_embeddings(
    model: PreTrainedModel,
    original_vocab_size: int,
    strategy: str,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    embed = model.get_input_embeddings()
    with torch.no_grad():
        if strategy == "mean":
            mean_vec = embed.weight[:original_vocab_size].mean(dim=0)
            embed.weight[original_vocab_size:] = mean_vec
        elif strategy == "clone_eos":
            eos_id = tokenizer.eos_token_id
            embed.weight[original_vocab_size:] = embed.weight[eos_id]
        # "random": resize_token_embeddings already initialises new rows randomly


def setup_gradient_masking(model: PreTrainedModel, original_vocab_size: int) -> None:
    """Freeze every parameter; allow gradients only on the new embedding rows.

    Uses a backward hook to zero out gradients for rows 0..original_vocab_size-1
    so that only the newly added rows receive updates during training.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    embed = model.get_input_embeddings()
    embed.weight.requires_grad_(True)

    def _mask_old_rows(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[:original_vocab_size] = 0
        return g

    embed.weight.register_hook(_mask_old_rows)
