from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    model_id: str
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class TokensConfig(BaseModel):
    num_tokens: int = Field(gt=0)
    token_prefix: str = "<|pref_"
    init_strategy: Literal["mean", "random", "clone_eos"] = "mean"


class DatasetConfig(BaseModel):
    path: str
    prompt_column: str = "prompt"
    preference_label_column: str = "preference_label"
    chosen_column: str = "chosen"
    rejected_column: Optional[str] = None
    max_length: int = 2048


class TrainingConfig(BaseModel):
    output_dir: str = "outputs"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    eval_strategy: Literal["steps", "epoch"] = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    contrastive_loss_weight: float = 0.0
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"


class InferenceConfig(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class AppConfig(BaseModel):
    model: ModelConfig
    tokens: TokensConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
