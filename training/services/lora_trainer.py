"""
LoRA fine-tuning service for Whisper.
Implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import DatasetDict

logger = logging.getLogger(__name__)


@dataclass
class WhisperLoRAConfig:
    """Configuration for LoRA fine-tuning"""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply LoRA to attention projection layers
            self.target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig"""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )


class DataCollatorForWhisper:
    """
    Data collator for Whisper training.
    Processes audio and text for batch training.
    """

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.

        Args:
            features: List of dicts with 'audio' and 'text' keys

        Returns:
            Dict with input_features and labels tensors
        """
        # Extract audio arrays and texts
        audio_arrays = [f['audio']['array'] for f in features]
        texts = [f['text'] for f in features]

        # Process audio to input features (log-mel spectrograms)
        input_features = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features

        # Tokenize text labels
        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids

        # Replace padding token id with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "labels": labels,
        }


class ProgressCallback(TrainerCallback):
    """Callback to track training progress"""

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs are available"""
        if self.progress_callback and logs:
            self.progress_callback(state.epoch, state.global_step, logs)


class WhisperLoRATrainer:
    """
    LoRA fine-tuning trainer for Whisper models.
    """

    def __init__(
        self,
        base_model: str = "openai/whisper-small",
        lora_config: WhisperLoRAConfig = None,
        output_dir: str = "./checkpoints",
        device: str = None,
    ):
        """
        Initialize trainer.

        Args:
            base_model: HuggingFace model ID (e.g., "openai/whisper-small")
            lora_config: LoRA configuration
            output_dir: Directory to save checkpoints
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.base_model_name = base_model
        self.lora_config = lora_config or WhisperLoRAConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load processor
        logger.info(f"Loading processor from: {base_model}")
        self.processor = WhisperProcessor.from_pretrained(base_model)

        # Model will be loaded in setup_model()
        self.model = None

    def setup_model(self) -> WhisperForConditionalGeneration:
        """
        Load base model and apply LoRA adapters.

        Returns:
            PEFT model with LoRA adapters
        """
        logger.info(f"Loading base model: {self.base_model_name}")

        # Load base Whisper model
        model = WhisperForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
        )

        # Disable caching for gradient checkpointing
        model.config.use_cache = False

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        # Apply LoRA
        logger.info(f"Applying LoRA with r={self.lora_config.r}, alpha={self.lora_config.lora_alpha}")
        peft_config = self.lora_config.to_peft_config()
        model = get_peft_model(model, peft_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / total_params

        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({trainable_pct:.2f}%)"
        )

        # Print trainable layers
        model.print_trainable_parameters()

        self.model = model
        return model

    def train(
        self,
        dataset: DatasetDict,
        epochs: int = 10,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: int = 500,
        early_stopping_patience: int = 5,
        progress_callback: Callable = None,
    ) -> str:
        """
        Train the model with LoRA.

        Args:
            dataset: HuggingFace DatasetDict with train/validation splits
            epochs: Number of training epochs
            batch_size: Per-device batch size
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for clipping
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            early_stopping_patience: Early stopping patience
            progress_callback: Optional callback for progress updates

        Returns:
            Path to best checkpoint
        """
        if self.model is None:
            self.setup_model()

        # Effective batch size
        effective_batch_size = batch_size * gradient_accumulation_steps
        logger.info(f"Effective batch size: {effective_batch_size}")

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),

            # Training
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            # Optimization
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            optim="adamw_torch",

            # Evaluation & Saving
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,  # Keep only 3 best checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Generation (for evaluation)
            predict_with_generate=True,
            generation_max_length=225,

            # Logging
            logging_steps=50,
            logging_dir=str(self.output_dir / "logs"),
            report_to=["tensorboard"],

            # Misc
            remove_unused_columns=False,
            label_names=["labels"],

            # Memory optimization
            gradient_checkpointing=True,
            fp16=False,  # Use float32 for stability on CPU
            dataloader_num_workers=0,  # Single worker for stability
        )

        # Data collator
        data_collator = DataCollatorForWhisper(self.processor)

        # Callbacks
        callbacks = []
        if progress_callback:
            callbacks.append(ProgressCallback(progress_callback))

        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting training...")
        logger.info(f"Total training samples: {len(dataset['train'])}")
        logger.info(f"Total validation samples: {len(dataset['validation'])}")

        train_result = trainer.train()

        # Log training metrics
        logger.info("Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")

        # Save best model
        best_checkpoint = str(self.output_dir / "best_model")
        trainer.save_model(best_checkpoint)
        self.processor.save_pretrained(best_checkpoint)

        logger.info(f"Best model saved to: {best_checkpoint}")

        # Save training metrics
        metrics_path = self.output_dir / "training_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)

        return best_checkpoint

    def merge_and_save(self, checkpoint_path: str, output_path: str) -> str:
        """
        Merge LoRA weights into base model and save as full model.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            output_path: Path to save merged model

        Returns:
            Path to merged model
        """
        logger.info("Merging LoRA weights into base model...")

        # Load base model
        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, checkpoint_path)

        # Merge weights
        logger.info("Merging weights...")
        merged_model = model.merge_and_unload()

        # Save merged model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        merged_model.save_pretrained(str(output_path))
        self.processor.save_pretrained(str(output_path))

        logger.info(f"Merged model saved to: {output_path}")

        return str(output_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a saved LoRA checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Load base model
        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
        )

        # Load PEFT model with LoRA weights
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)

        logger.info("Checkpoint loaded successfully")

    def evaluate(self, dataset: DatasetDict, split: str = 'test') -> Dict:
        """
        Evaluate model on a dataset split.

        Args:
            dataset: HuggingFace DatasetDict
            split: Split to evaluate on ('test', 'validation', etc.)

        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model() or load_checkpoint() first.")

        logger.info(f"Evaluating on {split} split...")

        # Create temporary trainer for evaluation
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir / "eval"),
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=10,
        )

        data_collator = DataCollatorForWhisper(self.processor)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
        )

        # Evaluate
        eval_results = trainer.evaluate(dataset[split])

        logger.info(f"Evaluation results: {eval_results}")

        return eval_results
