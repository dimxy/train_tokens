from __future__ import annotations

import torch
import pytest

from train_tokens.config import (
    AppConfig, DatasetConfig, InferenceConfig, ModelConfig, TokensConfig, TrainingConfig
)


# ── Minimal in-process model / tokenizer that need no downloads ───────────────

class FakeEmbedding(torch.nn.Embedding):
    pass


class FakeModel:
    """Minimal model stub with a real Embedding layer."""

    def __init__(self, vocab_size: int = 50, hidden_size: int = 32) -> None:
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._embed = FakeEmbedding(vocab_size, hidden_size)

    def get_input_embeddings(self) -> FakeEmbedding:
        return self._embed

    def resize_token_embeddings(self, new_size: int) -> None:
        new_embed = FakeEmbedding(new_size, self._hidden_size)
        with torch.no_grad():
            n = min(self._vocab_size, new_size)
            new_embed.weight.data[:n] = self._embed.weight.data[:n]
        self._embed = new_embed
        self._vocab_size = new_size

    def parameters(self):
        return self._embed.parameters()


class FakeTokenizer:
    """Minimal tokenizer stub with a deterministic character-level encoder
    and a stand-in chat template (`<role>content</role>` per turn)."""

    def __init__(self, vocab_size: int = 50) -> None:
        self._base_vocab_size = vocab_size
        self._extra: list[str] = []
        self.eos_token_id = 1
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.chat_template = "fake-template"

    def __len__(self) -> int:
        return self._base_vocab_size + len(self._extra)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        new = special_tokens_dict.get("additional_special_tokens", [])
        self._extra.extend(new)
        return len(new)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        idx = token_id - self._base_vocab_size
        return self._extra[idx]

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [(ord(c) % 48) + 2 for c in text]
        return type("Enc", (), {"input_ids": ids})()

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
    ):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        text = "".join(parts)
        if not tokenize:
            return text
        ids = self(text, add_special_tokens=False).input_ids
        if return_tensors == "pt":
            import torch
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return f"decoded:{list(token_ids)}"

    def save_pretrained(self, path: str) -> None:
        pass


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel(vocab_size=50, hidden_size=32)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer(vocab_size=50)


@pytest.fixture
def tokens_cfg() -> TokensConfig:
    return TokensConfig(num_tokens=3, token_prefix="", init_strategy="mean")


@pytest.fixture
def app_cfg(tmp_path) -> AppConfig:
    return AppConfig(
        model=ModelConfig(model_id="dummy/model"),
        tokens=TokensConfig(num_tokens=3, token_prefix=""),
        dataset=DatasetConfig(
            paths={"train": "dummy.jsonl"},
            messages_column="messages",
            max_length=256,
        ),
        training=TrainingConfig(output_dir=str(tmp_path / "out")),
        inference=InferenceConfig(),
    )
