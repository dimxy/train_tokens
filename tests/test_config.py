from __future__ import annotations

import pytest
from pydantic import ValidationError

from train_tokens.config import AppConfig, ModelConfig, TokensConfig, DatasetConfig, TrainingConfig


def test_from_yaml_round_trips(tmp_path):
    import yaml

    data = {
        "model": {"model_id": "Qwen/Qwen2-0.5B"},
        "tokens": {"num_tokens": 3},
        "dataset": {"path": "data.jsonl"},
        "training": {"output_dir": str(tmp_path)},
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump(data))

    cfg = AppConfig.from_yaml(cfg_file)
    assert cfg.model.model_id == "Qwen/Qwen2-0.5B"
    assert cfg.tokens.num_tokens == 3


def test_missing_model_id_raises():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({})


def test_missing_dataset_path_raises():
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({})


def test_num_tokens_must_be_positive():
    with pytest.raises(ValidationError):
        TokensConfig(num_tokens=0)

    with pytest.raises(ValidationError):
        TokensConfig(num_tokens=-1)


def test_invalid_dtype_raises():
    with pytest.raises(ValidationError):
        ModelConfig(model_id="x", dtype="int8")


def test_invalid_init_strategy_raises():
    with pytest.raises(ValidationError):
        TokensConfig(num_tokens=1, init_strategy="xavier")


def test_defaults_are_sensible():
    cfg = AppConfig(
        model=ModelConfig(model_id="m"),
        tokens=TokensConfig(num_tokens=1),
        dataset=DatasetConfig(path="p"),
    )
    assert cfg.training.seed == 42
    assert cfg.inference.do_sample is True
    assert cfg.tokens.init_strategy == "mean"
