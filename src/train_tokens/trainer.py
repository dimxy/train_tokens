from __future__ import annotations

from transformers import Trainer, TrainingArguments
import torch

from .config import TrainingConfig


class PreferenceTokenTrainer(Trainer):
    """Trainer that computes the standard causal-LM loss; only the new
    embedding rows have `requires_grad=True`, so gradients flow only there."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        return (out.loss, out) if return_outputs else out.loss


def make_training_args(cfg: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=cfg.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
