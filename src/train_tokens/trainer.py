from __future__ import annotations

from transformers import Trainer, TrainingArguments
import torch

from .config import TrainingConfig


class PreferenceTokenTrainer(Trainer):
    """Trainer that computes CLM loss on chosen responses with an optional
    contrastive term that increases loss on rejected responses.

    total_loss = chosen_loss - contrastive_weight * rejected_loss

    Minimising this maximises the gap between how well the model predicts
    chosen vs rejected responses given the same preference prefix token.
    contrastive_weight=0.0 (default) reduces to standard SFT.
    """

    def __init__(self, *args, contrastive_weight: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        chosen_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = chosen_out.loss

        if self.contrastive_weight > 0.0 and "rejected_input_ids" in inputs:
            rej_out = model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
                labels=inputs["rejected_labels"],
            )
            loss = loss - self.contrastive_weight * rej_out.loss

        return (loss, chosen_out) if return_outputs else loss


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
