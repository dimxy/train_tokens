from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="train-tokens", add_completion=False)

_DTYPE_MAP = {
    "bfloat16": None,  # populated lazily to avoid importing torch at module level
    "float16": None,
    "float32": None,
}


def _dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _load_model_and_tokenizer(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"torch_dtype": _dtype(cfg.model.dtype), "device_map": "auto"}
    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=_dtype(cfg.model.dtype)
        )
    elif cfg.model.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, **load_kwargs)
    return model, tokenizer


@app.command()
def train(config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config")):
    """Run the full training pipeline from a config file."""
    from transformers import set_seed

    from .checkpointing import save_artefacts
    from .config import AppConfig
    from .data import PreferenceCollator, build_processed_dataset, discover_labels, load_dataset
    from .tokenizer_utils import register_preference_tokens, setup_gradient_masking
    from .trainer import PreferenceTokenTrainer, make_training_args

    cfg = AppConfig.from_yaml(config)
    set_seed(cfg.training.seed)

    model, tokenizer = _load_model_and_tokenizer(cfg)

    train_ds = load_dataset(cfg, "train")
    labels = discover_labels(train_ds, cfg.dataset.preference_label_column)

    if len(labels) != cfg.tokens.num_tokens:
        typer.echo(
            f"Note: config.tokens.num_tokens={cfg.tokens.num_tokens} but dataset "
            f"has {len(labels)} unique labels {labels}. Using {len(labels)} tokens.",
            err=True,
        )

    label_to_token_id, original_vocab_size = register_preference_tokens(
        model, tokenizer, labels, cfg.tokens
    )
    setup_gradient_masking(model, original_vocab_size)

    train_processed = build_processed_dataset(train_ds, tokenizer, label_to_token_id, cfg)

    try:
        eval_ds = load_dataset(cfg, "validation")
        eval_processed = build_processed_dataset(eval_ds, tokenizer, label_to_token_id, cfg)
    except Exception:
        n = min(50, len(train_processed))
        eval_processed = train_processed.select(range(n))

    has_rejected = (
        cfg.dataset.rejected_column is not None
        and "rejected_input_ids" in train_processed.column_names
    )
    collator = PreferenceCollator(tokenizer=tokenizer, has_rejected=has_rejected)

    trainer = PreferenceTokenTrainer(
        model=model,
        args=make_training_args(cfg.training),
        train_dataset=train_processed,
        eval_dataset=eval_processed,
        data_collator=collator,
        contrastive_weight=cfg.training.contrastive_loss_weight,
    )
    trainer.train()

    save_artefacts(model, tokenizer, label_to_token_id, original_vocab_size, cfg.training.output_dir)
    typer.echo(f"Done. Artefacts saved to {cfg.training.output_dir}")


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c"),
    split: str = typer.Option("test", "--split", "-s"),
):
    """Compute eval loss on a dataset split using a trained checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .checkpointing import load_artefacts
    from .config import AppConfig
    from .data import PreferenceCollator, build_processed_dataset, load_dataset
    from .trainer import PreferenceTokenTrainer, make_training_args

    cfg = AppConfig.from_yaml(config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.training.output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        torch_dtype=_dtype(cfg.model.dtype),
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    label_to_token_id = load_artefacts(model, cfg.training.output_dir)

    ds = load_dataset(cfg, split)
    processed = build_processed_dataset(ds, tokenizer, label_to_token_id, cfg)

    has_rejected = (
        cfg.dataset.rejected_column is not None
        and "rejected_input_ids" in processed.column_names
    )
    collator = PreferenceCollator(tokenizer=tokenizer, has_rejected=has_rejected)

    trainer = PreferenceTokenTrainer(
        model=model,
        args=make_training_args(cfg.training),
        eval_dataset=processed,
        data_collator=collator,
    )
    metrics = trainer.evaluate()
    typer.echo(metrics)


@app.command(name="generate")
def generate_cmd(
    config: Path = typer.Option(..., "--config", "-c"),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    preference: str = typer.Option(..., "--preference", help="Preference label, e.g. 'formal'"),
):
    """Generate a response for a prompt steered by a trained preference token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .checkpointing import load_artefacts
    from .config import AppConfig
    from .inference import generate

    cfg = AppConfig.from_yaml(config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.training.output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        torch_dtype=_dtype(cfg.model.dtype),
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    label_to_token_id = load_artefacts(model, cfg.training.output_dir)

    response = generate(model, tokenizer, prompt, preference, label_to_token_id, cfg.inference)
    typer.echo(response)


if __name__ == "__main__":
    app()
