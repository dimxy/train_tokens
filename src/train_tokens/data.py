from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizerBase

from .config import AppConfig


def load_dataset(cfg: AppConfig, split: str) -> Dataset:
    path = cfg.dataset.path
    if any(path.endswith(ext) for ext in (".jsonl", ".json", ".jsonlines")):
        ds = hf_load_dataset("json", data_files={split: path}, split=split)
    else:
        ds = hf_load_dataset(path, split=split)

    required = [
        cfg.dataset.prompt_column,
        cfg.dataset.preference_label_column,
        cfg.dataset.chosen_column,
    ]
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return ds


def discover_labels(ds: Dataset, label_column: str) -> list[str]:
    """Return sorted unique preference labels found in the dataset."""
    return sorted({str(ex[label_column]) for ex in ds})


def build_processed_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    label_to_token_id: dict[str, int],
    cfg: AppConfig,
) -> Dataset:
    """Tokenize each example, prepend the preference token, and mask prompt loss.

    Output columns: input_ids, labels, and optionally rejected_input_ids /
    rejected_labels when cfg.dataset.rejected_column is set.
    """
    use_rejected = cfg.dataset.rejected_column is not None

    def _process(example: dict) -> dict:
        label = str(example[cfg.dataset.preference_label_column])
        if label not in label_to_token_id:
            raise ValueError(
                f"Label '{label}' not in token mapping. Known: {list(label_to_token_id)}"
            )
        pref_ids = [label_to_token_id[label]]

        prompt_ids = tokenizer(
            example[cfg.dataset.prompt_column], add_special_tokens=False
        ).input_ids
        chosen_ids = tokenizer(
            example[cfg.dataset.chosen_column], add_special_tokens=False
        ).input_ids

        # Loss is computed only on the response tokens; prefix + prompt are masked.
        prefix_len = len(pref_ids) + len(prompt_ids)
        input_ids = pref_ids + prompt_ids + chosen_ids + [tokenizer.eos_token_id]
        labels = [-100] * prefix_len + chosen_ids + [tokenizer.eos_token_id]

        max_len = cfg.dataset.max_length
        result = {
            "input_ids": input_ids[:max_len],
            "labels": labels[:max_len],
        }

        if use_rejected:
            rejected_text = example.get(cfg.dataset.rejected_column)
            if rejected_text:
                rej_ids = tokenizer(rejected_text, add_special_tokens=False).input_ids
                rej_input = pref_ids + prompt_ids + rej_ids + [tokenizer.eos_token_id]
                rej_labels = [-100] * prefix_len + rej_ids + [tokenizer.eos_token_id]
                result["rejected_input_ids"] = rej_input[:max_len]
                result["rejected_labels"] = rej_labels[:max_len]

        return result

    return ds.map(_process, remove_columns=ds.column_names)


@dataclass
class PreferenceCollator:
    tokenizer: PreTrainedTokenizerBase
    has_rejected: bool = False

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = self._pad("input_ids", "labels", "attention_mask", features)
        if self.has_rejected and "rejected_input_ids" in features[0]:
            batch.update(
                self._pad(
                    "rejected_input_ids",
                    "rejected_labels",
                    "rejected_attention_mask",
                    features,
                )
            )
        return batch

    def _pad(
        self,
        ids_key: str,
        labels_key: str,
        mask_key: str,
        features: list[dict],
    ) -> dict[str, torch.Tensor]:
        seqs = [f[ids_key] for f in features]
        labs = [f[labels_key] for f in features]
        max_len = max(len(s) for s in seqs)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_ids, padded_labs, masks = [], [], []
        for seq, lab in zip(seqs, labs):
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [pad_id] * pad_len)
            padded_labs.append(lab + [-100] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)

        return {
            ids_key: torch.tensor(padded_ids, dtype=torch.long),
            labels_key: torch.tensor(padded_labs, dtype=torch.long),
            mask_key: torch.tensor(masks, dtype=torch.long),
        }
