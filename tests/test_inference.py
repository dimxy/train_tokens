from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import pytest

from train_tokens.config import InferenceConfig
from train_tokens.inference import generate


LABEL_TO_TOKEN_ID = {"formal": 50, "concise": 51}


@pytest.fixture
def inf_cfg():
    return InferenceConfig(max_new_tokens=10, temperature=1.0, top_p=1.0, do_sample=False)


def _make_model_mock(input_len: int, response_token_ids: list[int]):
    model = MagicMock()
    model.device = torch.device("cpu")
    full_output = torch.tensor([list(range(input_len)) + response_token_ids])
    model.generate.return_value = full_output
    return model


def _make_tokenizer_mock(prompt_ids: list[int]):
    tok = MagicMock()
    tok.eos_token_id = 1

    def call(text, add_special_tokens=True):
        return type("Enc", (), {"input_ids": prompt_ids})()

    tok.side_effect = call
    tok.__call__ = call
    tok.decode.return_value = "mocked response"
    return tok


def test_generate_prepends_preference_token(inf_cfg):
    prompt_ids = [10, 11, 12]
    model = _make_model_mock(input_len=1 + len(prompt_ids), response_token_ids=[20, 21])
    tokenizer = _make_tokenizer_mock(prompt_ids)

    generate(model, tokenizer, "hello", "formal", LABEL_TO_TOKEN_ID, inf_cfg)

    call_args = model.generate.call_args
    input_ids_tensor = call_args[0][0]
    assert input_ids_tensor[0, 0].item() == LABEL_TO_TOKEN_ID["formal"]
    assert input_ids_tensor[0, 1:].tolist() == prompt_ids


def test_generate_decodes_only_new_tokens(inf_cfg):
    prompt_ids = [10, 11]
    response_ids = [30, 31, 32]
    model = _make_model_mock(input_len=1 + len(prompt_ids), response_token_ids=response_ids)
    tokenizer = _make_tokenizer_mock(prompt_ids)

    generate(model, tokenizer, "hi", "concise", LABEL_TO_TOKEN_ID, inf_cfg)

    decoded_ids = tokenizer.decode.call_args[0][0].tolist()
    assert decoded_ids == response_ids


def test_unknown_preference_label_raises(inf_cfg):
    model = MagicMock()
    tokenizer = MagicMock()
    with pytest.raises(ValueError, match="Unknown preference label"):
        generate(model, tokenizer, "test", "nonexistent", LABEL_TO_TOKEN_ID, inf_cfg)
