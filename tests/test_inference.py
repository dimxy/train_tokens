from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest

from train_tokens.config import InferenceConfig
from train_tokens.inference import generate


LABEL_TO_TOKEN_ID = {"formal": 50, "concise": 51}


@pytest.fixture
def inf_cfg():
    return InferenceConfig(max_new_tokens=10, temperature=1.0, top_p=1.0, do_sample=False)


def _make_tokenizer_mock(prompt_input_ids: list[int], pref_token_text: str):
    tok = MagicMock()
    tok.eos_token_id = 1
    tok.chat_template = "fake-template"
    tok.convert_ids_to_tokens.return_value = pref_token_text
    tok.apply_chat_template.return_value = "rendered chat text"

    encoding = MagicMock()
    encoding.input_ids = torch.tensor([prompt_input_ids], dtype=torch.long)
    encoding.attention_mask = torch.ones((1, len(prompt_input_ids)), dtype=torch.long)
    tok.return_value = encoding
    tok.decode.return_value = "mocked response"
    return tok


def _make_model_mock(input_len: int, response_token_ids: list[int]):
    model = MagicMock()
    model.device = torch.device("cpu")
    full_output = torch.tensor([list(range(input_len)) + response_token_ids])
    model.generate.return_value = full_output
    return model


def test_generate_embeds_preference_token_in_user_content(inf_cfg):
    prompt_ids = [10, 11, 12, 13]
    tokenizer = _make_tokenizer_mock(prompt_ids, pref_token_text="<|formal|>")
    model = _make_model_mock(input_len=len(prompt_ids), response_token_ids=[20, 21])

    generate(model, tokenizer, "hello", "formal", LABEL_TO_TOKEN_ID, inf_cfg)

    messages = tokenizer.apply_chat_template.call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert user_msg["content"].startswith("<|formal|>")
    assert "hello" in user_msg["content"]
    kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs.get("add_generation_prompt") is True


def test_generate_decodes_only_new_tokens(inf_cfg):
    prompt_ids = [10, 11]
    response_ids = [30, 31, 32]
    tokenizer = _make_tokenizer_mock(prompt_ids, pref_token_text="<|concise|>")
    model = _make_model_mock(input_len=len(prompt_ids), response_token_ids=response_ids)

    generate(model, tokenizer, "hi", "concise", LABEL_TO_TOKEN_ID, inf_cfg)

    decoded_ids = tokenizer.decode.call_args[0][0].tolist()
    assert decoded_ids == response_ids


def test_unknown_preference_label_raises(inf_cfg):
    model = MagicMock()
    tokenizer = MagicMock()
    with pytest.raises(ValueError, match="Unknown preference label"):
        generate(model, tokenizer, "test", "nonexistent", LABEL_TO_TOKEN_ID, inf_cfg)
