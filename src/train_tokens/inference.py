from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import InferenceConfig


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    preference_label: str,
    label_to_token_id: dict[str, int],
    cfg: InferenceConfig,
) -> str:
    if preference_label not in label_to_token_id:
        raise ValueError(
            f"Unknown preference label '{preference_label}'. "
            f"Known labels: {sorted(label_to_token_id)}"
        )

    pref_token_id = label_to_token_id[preference_label]
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = torch.tensor(
        [[pref_token_id] + prompt_ids], dtype=torch.long, device=model.device
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
