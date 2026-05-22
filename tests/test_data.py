from __future__ import annotations

import pytest
from datasets import Dataset

from train_tokens.data import build_processed_dataset, discover_labels, PreferenceCollator


LABEL_TO_TOKEN_ID = {"concise": 50, "formal": 51}

RAW_EXAMPLES = [
    {"prompt": "Hello", "preference_label": "formal", "chosen": "Good day"},
    {"prompt": "Hi", "preference_label": "concise", "chosen": "Hey"},
    {"prompt": "Thanks", "preference_label": "formal", "chosen": "Many thanks"},
]


@pytest.fixture
def raw_ds() -> Dataset:
    return Dataset.from_list(RAW_EXAMPLES)


def test_discover_labels_returns_sorted_unique(raw_ds):
    labels = discover_labels(raw_ds, "preference_label")
    assert labels == sorted({"formal", "concise"})


def test_augmented_input_ids_start_with_pref_token(fake_tokenizer, app_cfg, raw_ds):
    processed = build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)
    for ex in processed:
        label = RAW_EXAMPLES[processed.to_list().index(ex)]["preference_label"]
        expected_token = LABEL_TO_TOKEN_ID[label]
        assert ex["input_ids"][0] == expected_token, "First token must be the preference token"


def test_labels_mask_prompt_portion(fake_tokenizer, app_cfg, raw_ds):
    processed = build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)
    for ex in processed:
        labels = ex["labels"]
        # At least the first element (pref token position) must be masked
        assert labels[0] == -100
        # At least one label must be non-masked (the response)
        assert any(l != -100 for l in labels)


def test_unknown_label_raises(fake_tokenizer, app_cfg):
    bad_ds = Dataset.from_list([
        {"prompt": "X", "preference_label": "unknown_label", "chosen": "Y"}
    ])
    with pytest.raises(ValueError, match="not in token mapping"):
        build_processed_dataset(bad_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)


def test_output_does_not_exceed_max_length(fake_tokenizer, app_cfg, raw_ds):
    processed = build_processed_dataset(raw_ds, fake_tokenizer, LABEL_TO_TOKEN_ID, app_cfg)
    for ex in processed:
        assert len(ex["input_ids"]) <= app_cfg.dataset.max_length
        assert len(ex["labels"]) <= app_cfg.dataset.max_length


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
    # Shorter sequence must be right-padded
    assert batch["attention_mask"][0, 3].item() == 0
    assert batch["attention_mask"][0, 4].item() == 0


def test_collator_rejected_fields_included_when_present(fake_tokenizer):
    features = [
        {
            "input_ids": [1, 2],
            "labels": [-100, 3],
            "rejected_input_ids": [1, 4, 5],
            "rejected_labels": [-100, 4, 5],
        }
    ]
    collator = PreferenceCollator(tokenizer=fake_tokenizer, has_rejected=True)
    batch = collator(features)

    assert "rejected_input_ids" in batch
    assert "rejected_attention_mask" in batch
    assert "rejected_labels" in batch
