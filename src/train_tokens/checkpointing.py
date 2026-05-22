from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def save_artefacts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    label_to_token_id: dict[str, int],
    original_vocab_size: int,
    save_dir: str | Path,
) -> None:
    """Persist only the new embedding rows and their label mapping.

    The base model weights are not saved; artefacts are portable across
    any checkpoint of the same Qwen2 variant.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    embed = model.get_input_embeddings()
    new_embeddings = embed.weight[original_vocab_size:].detach().cpu()
    torch.save(new_embeddings, save_dir / "preference_embeddings.pt")

    config = {
        "label_to_token_id": label_to_token_id,
        "original_vocab_size": original_vocab_size,
        "num_tokens": len(label_to_token_id),
        "hidden_size": int(new_embeddings.shape[-1]),
    }
    (save_dir / "token_config.json").write_text(json.dumps(config, indent=2))
    tokenizer.save_pretrained(save_dir)


def load_artefacts(
    model: PreTrainedModel,
    save_dir: str | Path,
) -> dict[str, int]:
    """Load saved embedding rows back into model and return the label mapping."""
    save_dir = Path(save_dir)
    config = json.loads((save_dir / "token_config.json").read_text())

    embed = model.get_input_embeddings()
    new_embeddings = torch.load(
        save_dir / "preference_embeddings.pt",
        map_location=embed.weight.device,
        weights_only=True,
    )
    with torch.no_grad():
        embed.weight[config["original_vocab_size"] :] = new_embeddings

    return config["label_to_token_id"]
