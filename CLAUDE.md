# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Create venv (Python 3.11 required — torch has no macOS x86_64 wheels for 3.12+)
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]"

# Run all unit tests (no model download needed)
PYTHONPATH=src .venv/bin/pytest tests/ -q

# Run a single test file
PYTHONPATH=src .venv/bin/pytest tests/test_tokenizer_utils.py -v

# Train
.venv/bin/python -m train_tokens.cli train --config configs/default.yaml

# Evaluate on test split
.venv/bin/python -m train_tokens.cli evaluate --config configs/default.yaml --split test

# Generate with a trained checkpoint (preference must match a label seen during training)
.venv/bin/python -m train_tokens.cli generate \
  --config configs/default.yaml \
  --prompt "Summarise this article…" \
  --preference humorous
```

## What is being built

A Python application that adds learnable special tokens (one per preference label, e.g. `<|humorous|>`) to a Qwen2 language model and trains only those token embeddings on a conversational preference dataset, so the label token can be embedded in a user message at inference time to steer model responses toward user-defined preferences (e.g. formal, concise, humorous). The base model weights are never updated — this is prompt tuning over new vocabulary entries.

## Planned project layout (from requirements.yaml TR-05)

```
src/train_tokens/
  config.py           # Pydantic models for the YAML config schema
  tokenizer_utils.py  # token registration (FR-01)
  data.py             # dataset loading and input construction (FR-02, FR-03)
  trainer.py          # training loop (FR-04)
  checkpointing.py    # save/load artefacts (FR-05)
  inference.py        # inference helper (FR-06)
  cli.py              # Typer CLI: train / evaluate / generate
configs/
  default.yaml        # all tuneable values; no magic constants in source
tests/
```

## Key design decisions (do not change without updating requirements.yaml)

- **Approach**: prompt tuning — only the new embedding rows (shape `[num_tokens, hidden_size]`) have `requires_grad=True`; all other model parameters are frozen.
- **Token initialisation**: new embeddings start from the mean of existing embeddings (not random).
- **Dataset format**: each row is a chat transcript: `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}`. The preference token (e.g. `<|humorous|>`) is embedded inline inside a user message; once registered as a special token it tokenises to a single id. Each conversation must end with an assistant turn — that turn is the only span the loss is computed over.
- **Training input construction**: feed the full `messages` list through `tokenizer.apply_chat_template(..., tokenize=True)`. Mask labels to -100 for everything up to the assistant header (computed by re-applying the template to `messages[:-1]` with `add_generation_prompt=True`). The preference token **must** appear in the input during training — it is the sole gradient path to the new embeddings.
- **Saved artefact**: `preference_embeddings.pt` + `token_config.json` only — independent of base model weights, portable across checkpoints of the same Qwen2 variant.
- **Config**: every tuneable value lives in a single YAML file validated by Pydantic; no hardcoded constants in source.

## Requirements spec

`requirements.yaml` defines FR-01 through FR-06, TR-01 through TR-06, DR-01/DR-02, NFR-01 through NFR-05, and TEST-01 through TEST-03. Cross-reference by ID when implementing or reviewing code.
