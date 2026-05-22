from __future__ import annotations

import torch
import pytest

from train_tokens.config import TokensConfig
from train_tokens.tokenizer_utils import register_preference_tokens, setup_gradient_masking


LABELS = ["concise", "empathetic", "formal"]  # 3 labels, sorted alphabetically


def test_token_ids_are_contiguous(fake_model, fake_tokenizer, tokens_cfg):
    original_size = len(fake_tokenizer)
    label_to_token_id, returned_orig = register_preference_tokens(
        fake_model, fake_tokenizer, LABELS, tokens_cfg
    )
    assert returned_orig == original_size
    ids = sorted(label_to_token_id.values())
    assert ids == list(range(original_size, original_size + len(LABELS)))


def test_ids_do_not_collide_with_existing(fake_model, fake_tokenizer, tokens_cfg):
    original_size = len(fake_tokenizer)
    label_to_token_id, _ = register_preference_tokens(
        fake_model, fake_tokenizer, LABELS, tokens_cfg
    )
    existing_ids = set(range(original_size))
    assert not existing_ids.intersection(label_to_token_id.values())


def test_vocab_size_increases_by_num_labels(fake_model, fake_tokenizer, tokens_cfg):
    original_size = len(fake_tokenizer)
    register_preference_tokens(fake_model, fake_tokenizer, LABELS, tokens_cfg)
    assert len(fake_tokenizer) == original_size + len(LABELS)


def test_mean_init_matches_existing_mean(fake_model, fake_tokenizer):
    original_size = len(fake_tokenizer)
    original_mean = fake_model.get_input_embeddings().weight.data[:original_size].mean(dim=0)

    cfg = TokensConfig(num_tokens=1, token_prefix="<|pref_", init_strategy="mean")
    register_preference_tokens(fake_model, fake_tokenizer, ["x"], cfg)

    new_row = fake_model.get_input_embeddings().weight.data[original_size]
    assert torch.allclose(new_row, original_mean, atol=1e-6)


def test_embedding_matrix_row_count_grows(fake_model, fake_tokenizer, tokens_cfg):
    original_size = len(fake_tokenizer)
    register_preference_tokens(fake_model, fake_tokenizer, LABELS, tokens_cfg)
    embed = fake_model.get_input_embeddings()
    assert embed.weight.shape[0] == original_size + len(LABELS)


def test_embedding_hidden_dim_unchanged(fake_model, fake_tokenizer, tokens_cfg):
    hidden = fake_model.get_input_embeddings().weight.shape[1]
    register_preference_tokens(fake_model, fake_tokenizer, LABELS, tokens_cfg)
    assert fake_model.get_input_embeddings().weight.shape[1] == hidden


def test_gradient_masking_zeroes_old_rows(fake_model, fake_tokenizer, tokens_cfg):
    _, orig_size = register_preference_tokens(fake_model, fake_tokenizer, LABELS, tokens_cfg)
    setup_gradient_masking(fake_model, orig_size)

    embed = fake_model.get_input_embeddings()
    embed.weight.sum().backward()

    assert embed.weight.grad is not None
    assert torch.all(embed.weight.grad[:orig_size] == 0), "Old rows must have zero gradient"
    assert not torch.all(embed.weight.grad[orig_size:] == 0), "New rows must receive gradient"


def test_gradient_masking_freezes_all_other_params(fake_model, fake_tokenizer, tokens_cfg):
    _, orig_size = register_preference_tokens(fake_model, fake_tokenizer, LABELS, tokens_cfg)
    setup_gradient_masking(fake_model, orig_size)

    trainable = [p for p in fake_model.parameters() if p.requires_grad]
    assert len(trainable) == 1
    assert trainable[0] is fake_model.get_input_embeddings().weight
