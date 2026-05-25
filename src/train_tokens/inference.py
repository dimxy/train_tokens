from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import InferenceConfig
from .data import assert_chat_template

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    preference_label: str,
    label_to_token_id: dict[str, int],
    cfg: InferenceConfig,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    if preference_label not in label_to_token_id:
        raise ValueError(
            f"Unknown preference label '{preference_label}'. "
            f"Known labels: {sorted(label_to_token_id)}"
        )
    assert_chat_template(tokenizer)

    pref_token_id = label_to_token_id[preference_label]
    pref_token_text = tokenizer.convert_ids_to_tokens(pref_token_id)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{pref_token_text} {prompt}"},
    ]
    print('messages=', messages)
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoding = tokenizer(
        chat_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoding.input_ids.to(model.device)
    attention_mask = encoding.attention_mask.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
