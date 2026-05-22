from __future__ import annotations

import json

import torch
import pytest

from train_tokens.config import TokensConfig
from train_tokens.tokenizer_utils import register_preference_tokens
from train_tokens.checkpointing import save_artefacts, load_artefacts


LABELS = ["concise", "formal"]


@pytest.fixture
def trained_model_and_mapping(fake_model, fake_tokenizer):
    cfg = TokensConfig(num_tokens=2, token_prefix="<|pref_", init_strategy="mean")
    label_to_token_id, original_vocab_size = register_preference_tokens(
        fake_model, fake_tokenizer, LABELS, cfg
    )
    # Simulate training: assign distinct values to the new rows
    embed = fake_model.get_input_embeddings()
    with torch.no_grad():
        embed.weight[original_vocab_size] = 1.0
        embed.weight[original_vocab_size + 1] = 2.0

    return fake_model, fake_tokenizer, label_to_token_id, original_vocab_size


def test_save_creates_expected_files(trained_model_and_mapping, tmp_path):
    model, tokenizer, label_to_token_id, original_vocab_size = trained_model_and_mapping
    save_artefacts(model, tokenizer, label_to_token_id, original_vocab_size, tmp_path)

    assert (tmp_path / "preference_embeddings.pt").exists()
    assert (tmp_path / "token_config.json").exists()


def test_token_config_json_content(trained_model_and_mapping, tmp_path):
    model, tokenizer, label_to_token_id, original_vocab_size = trained_model_and_mapping
    save_artefacts(model, tokenizer, label_to_token_id, original_vocab_size, tmp_path)

    config = json.loads((tmp_path / "token_config.json").read_text())
    assert config["label_to_token_id"] == label_to_token_id
    assert config["original_vocab_size"] == original_vocab_size
    assert config["num_tokens"] == len(LABELS)


def test_load_restores_embeddings_exactly(trained_model_and_mapping, tmp_path):
    model, tokenizer, label_to_token_id, original_vocab_size = trained_model_and_mapping
    embed = model.get_input_embeddings()
    saved_new_rows = embed.weight[original_vocab_size:].detach().clone()

    save_artefacts(model, tokenizer, label_to_token_id, original_vocab_size, tmp_path)

    # Corrupt the new rows
    with torch.no_grad():
        embed.weight[original_vocab_size:] = 99.0

    loaded_mapping = load_artefacts(model, tmp_path)

    restored = model.get_input_embeddings().weight[original_vocab_size:].detach()
    assert torch.allclose(restored, saved_new_rows, atol=1e-6)
    assert loaded_mapping == label_to_token_id


def test_load_does_not_touch_old_rows(trained_model_and_mapping, tmp_path):
    model, tokenizer, label_to_token_id, original_vocab_size = trained_model_and_mapping
    embed = model.get_input_embeddings()
    old_rows_before = embed.weight[:original_vocab_size].detach().clone()

    save_artefacts(model, tokenizer, label_to_token_id, original_vocab_size, tmp_path)
    load_artefacts(model, tmp_path)

    old_rows_after = model.get_input_embeddings().weight[:original_vocab_size].detach()
    assert torch.allclose(old_rows_after, old_rows_before, atol=1e-6)
