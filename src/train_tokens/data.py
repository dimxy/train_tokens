from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizerBase

from .config import AppConfig

_TOKEN_PATTERN = re.compile(r"<\|([^|<>]+)\|>")


def load_dataset(cfg: AppConfig, split: str) -> Dataset:
    path = cfg.dataset.path
    if any(path.endswith(ext) for ext in (".jsonl", ".json", ".jsonlines")):
        ds = hf_load_dataset("json", data_files={split: path}, split=split)
    else:
        ds = hf_load_dataset(path, split=split)

    if cfg.dataset.messages_column not in ds.column_names:
        raise ValueError(
            f"Dataset missing required column: {cfg.dataset.messages_column}"
        )

    return ds


def discover_labels(ds: Dataset, messages_column: str) -> list[str]:
    """Return sorted unique preference labels parsed from user messages.

    Scans every `messages[*]` entry with `role == "user"` for `<|label|>`
    occurrences in its content and returns the sorted unique set.
    """
    found: set[str] = set()
    for ex in ds:
        for msg in ex[messages_column]:
            if msg.get("role") == "user":
                found.update(_TOKEN_PATTERN.findall(str(msg.get("content", ""))))
    return sorted(found)


def _last_assistant_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    raise ValueError("Conversation has no assistant message to train on")


def assert_chat_template(tokenizer: PreTrainedTokenizerBase) -> None:
    """Fail fast if the tokenizer has no chat template.

    Base Qwen2 checkpoints (e.g. Qwen/Qwen2-0.5B) ship without one; calling
    `apply_chat_template` on them silently returns the raw message text, which
    later surfaces as a cryptic `torch.tensor` "too many dimensions 'str'" error
    inside the data collator.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer has no chat_template. Base Qwen2 models "
            "(Qwen/Qwen2-0.5B, 1.5B, 7B) don't ship one — either switch to the "
            "matching -Instruct variant, or assign tokenizer.chat_template to "
            "the standard Qwen2 ChatML template before training."
        )


def build_processed_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    label_to_token_id: dict[str, int],
    cfg: AppConfig,
) -> Dataset:
    """Apply the chat template and mask loss outside the final assistant turn.

    The preference token (e.g. `<|humorous|>`) is expected inline in a user
    message; because it was registered as a special token it tokenises to a
    single id, which is the sole gradient path back to the new embedding rows.
    """
    assert_chat_template(tokenizer)
    messages_col = cfg.dataset.messages_column
    max_len = cfg.dataset.max_length

    def _process(example: dict) -> dict:
        messages = list(example[messages_col])
        last_asst = _last_assistant_index(messages)

        for msg in messages[:last_asst]:
            if msg.get("role") == "user":
                for lbl in _TOKEN_PATTERN.findall(str(msg.get("content", ""))):
                    if lbl not in label_to_token_id:
                        raise ValueError(
                            f"Label '{lbl}' not in token mapping. "
                            f"Known: {list(label_to_token_id)}"
                        )

        prefix_text = tokenizer.apply_chat_template(
            messages[:last_asst],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]

        return {
            "input_ids": full_ids[:max_len],
            "labels": labels[:max_len],
        }

    return ds.map(_process, remove_columns=ds.column_names)


@dataclass
class PreferenceCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [f["input_ids"] for f in features]
        labs = [f["labels"] for f in features]
        max_len = max(len(s) for s in seqs)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_ids, padded_labs, masks = [], [], []
        for seq, lab in zip(seqs, labs):
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [pad_id] * pad_len)
            padded_labs.append(lab + [-100] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
