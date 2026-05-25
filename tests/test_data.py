from __future__ import annotations

import pytest
from datasets import Dataset

from train_tokens.data import (
    assert_chat_template,
    build_processed_dataset,
    discover_labels,
    PreferenceCollator,
)


LABEL_TO_TOKEN_ID = {"concise": 50, "formal": 51, "humorous": 52}


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


RAW_EXAMPLES = [
    {"messages": [
        _msg("system", "You are a helpful assistant."),
        _msg("user", "<|formal|> Hello, who are you?"),
        _msg("assistant", "Good day. I am an assistant."),
    ]},
    {"messages": [
        _msg("system", "You are a helpful assistant."),
        _msg("user", "<|concise|> Define entropy."),
        _msg("assistant", "Disorder measure."),
    ]},
    {"messages": [
        _msg("system", "You are a helpful assistant."),
        _msg("user", "<|humorous|> How do I fix a merge conflict?"),
        _msg("assistant", "Open the file, pick a side, delete the markers, commit."),
    ]},
]


@pytest.fixture
def raw_ds() -> Dataset:
    return Dataset.from_list(RAW_EXAMPLES)


def test_discover_labels_returns_sorted_unique(raw_ds):
    labels = discover_labels(raw_ds, "messages")
    assert labels == ["concise", "formal", "humorous"]


def test_discover_labels_ignores_non_user_roles():
    ds = Dataset.from_list([
        {"messages": [
            _msg("system", "<|ignore_me|>"),
            _msg("user", "<|wanted|> hi"),
            _msg("assistant", "<|also_ignored|> hello"),
        ]},
    ])
    assert discover_labels(ds, "messages") == ["wanted"]


def test_input_ids_and_labels_have_equal_length(fake_tokenizer, app_cfg, raw_ds):
    processed = build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)
    for ex in processed:
        assert len(ex["input_ids"]) == len(ex["labels"])


def test_labels_mask_everything_before_assistant(fake_tokenizer, app_cfg, raw_ds):
    processed = build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)
    for ex in processed:
        labels = ex["labels"]
        assert labels[0] == -100, "First token (chat-template prefix) must be masked"
        assert any(l != -100 for l in labels), "Assistant response must contribute to loss"
        first_unmasked = next(i for i, l in enumerate(labels) if l != -100)
        assert all(l == -100 for l in labels[:first_unmasked])
        assert all(l != -100 for l in labels[first_unmasked:])


def test_assistant_response_tokens_match_input_ids(fake_tokenizer, app_cfg, raw_ds):
    processed = build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)
    for ex in processed:
        for ids, lab in zip(ex["input_ids"], ex["labels"]):
            assert lab in (-100, ids)


def test_unknown_label_raises(fake_tokenizer, app_cfg):
    bad_ds = Dataset.from_list([
        {"messages": [
            _msg("user", "<|unknown_label|> hi"),
            _msg("assistant", "hello"),
        ]},
    ])
    with pytest.raises(ValueError, match="not in token mapping"):
        build_processed_dataset(bad_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)


def test_missing_assistant_raises(fake_tokenizer, app_cfg):
    bad_ds = Dataset.from_list([
        {"messages": [_msg("user", "<|formal|> hi")]},
    ])
    with pytest.raises(ValueError, match="no assistant message"):
        build_processed_dataset(bad_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)


def test_output_does_not_exceed_max_length(fake_tokenizer, app_cfg, raw_ds):
    processed = build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)
    for ex in processed:
        assert len(ex["input_ids"]) <= app_cfg.dataset.max_length
        assert len(ex["labels"]) <= app_cfg.dataset.max_length


def test_missing_chat_template_raises(fake_tokenizer, app_cfg, raw_ds):
    fake_tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat_template"):
        build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)


def test_assert_chat_template_accepts_set_template(fake_tokenizer):
    assert_chat_template(fake_tokenizer)  # default fixture has one


def test_collator_pads_to_longest(fake_tokenizer):
    features = [
        {"input_ids": [1, 2, 3], "labels": [-100, 5, 6]},
        {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 7, 8, 9]},
    ]
    collator = PreferenceCollator(tokenizer=fake_tokenizer)
    batch = collator(features)

    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["attention_mask"][0, 3].item() == 0
    assert batch["attention_mask"][0, 4].item() == 0
