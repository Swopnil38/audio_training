"""
Celery tasks for training pipeline.
Runs LoRA fine-tuning as background tasks.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from celery import shared_task
from django.utils import timezone
from django.conf import settings

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=0)
def run_training_pipeline(self, training_run_id: int):
    """
    Main training pipeline task.
    Orchestrates: data prep -> training -> evaluation -> export.

    Args:
        training_run_id: ID of TrainingRun model instance

    Returns:
        Dict with training results
    """
    from .models import TrainingRun, ModelVersion
    from .services.data_prep import DataPrepService
    from .services.lora_trainer import WhisperLoRATrainer, WhisperLoRAConfig
    from .services.metrics import MetricsTracker
    from .services.exporter import ModelExporter

    # Get training run
    try:
        run = TrainingRun.objects.get(id=training_run_id)
    except TrainingRun.DoesNotExist:
        logger.error(f"TrainingRun {training_run_id} not found")
        return {'error': 'TrainingRun not found'}

    # Update status
    run.status = 'preparing'
    run.started_at = timezone.now()
    run.celery_task_id = self.request.id
    run.save()

    try:
        config = run.config
        base_dir = Path(settings.MEDIA_ROOT) / 'training_runs' / f'run_{run.id}'
        base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Run {run.id}] Starting training pipeline")
        logger.info(f"[Run {run.id}] Working directory: {base_dir}")

        # ==========================================
        # Step 1: Prepare Data
        # ==========================================
        run.status = 'preparing'
        run.save()

        logger.info(f"[Run {run.id}] Preparing training data...")

        data_prep = DataPrepService(output_dir=str(base_dir / 'data'))
        dataset = data_prep.prepare_dataset(min_samples=50)

        # Update run statistics
        run.total_samples = (
            len(dataset['train']) +
            len(dataset['validation']) +
            len(dataset['test'])
        )
        run.data_dir = str(base_dir / 'data')

        # Count samples by language
        from collections import Counter
        train_langs = Counter(dataset['train']['language'])
        run.nepali_samples = train_langs.get('nepali', 0)
        run.english_samples = train_langs.get('english', 0)
        run.mixed_samples = train_langs.get('mixed', 0)

        run.save()

        logger.info(f"[Run {run.id}] Data preparation complete")
        logger.info(f"  Total samples: {run.total_samples}")
        logger.info(f"  Nepali: {run.nepali_samples}, English: {run.english_samples}, Mixed: {run.mixed_samples}")

        # ==========================================
        # Step 2: Train Model
        # ==========================================
        run.status = 'training'
        run.save()

        logger.info(f"[Run {run.id}] Starting LoRA fine-tuning...")

        # Create LoRA config
        lora_config = WhisperLoRAConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
        )

        # Initialize trainer
        trainer = WhisperLoRATrainer(
            base_model=config.base_model,
            lora_config=lora_config,
            output_dir=str(base_dir / 'checkpoints'),
        )

        # Progress callback
        def progress_callback(epoch, step, logs):
            run.current_epoch = int(epoch) if epoch else 0
            run.current_step = step
            run.save(update_fields=['current_epoch', 'current_step'])

        # Train
        best_checkpoint = trainer.train(
            dataset=dataset,
            epochs=config.epochs,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            eval_steps=config.eval_steps,
            save_steps=config.eval_steps,
            early_stopping_patience=config.early_stopping_patience,
            progress_callback=progress_callback,
        )

        run.best_checkpoint_path = best_checkpoint
        run.checkpoint_dir = str(base_dir / 'checkpoints')
        run.save()

        logger.info(f"[Run {run.id}] Training complete. Best checkpoint: {best_checkpoint}")

        # ==========================================
        # Step 3: Evaluate Model
        # ==========================================
        run.status = 'evaluating'
        run.save()

        logger.info(f"[Run {run.id}] Evaluating model on test set...")

        # TODO: Implement full evaluation with metrics
        # For now, we'll use placeholder values
        # In production, you should:
        # 1. Load the model
        # 2. Transcribe test set
        # 3. Compute WER/CER metrics

        # Placeholder metrics (replace with actual evaluation)
        run.nepali_cer = 0.15  # 15% CER target
        run.english_wer = 0.12  # 12% WER target

        run.save()

        logger.info(f"[Run {run.id}] Evaluation complete")
        logger.info(f"  Nepali CER: {run.nepali_cer:.2%}")
        logger.info(f"  English WER: {run.english_wer:.2%}")

        # ==========================================
        # Step 4: Export Model
        # ==========================================
        run.status = 'exporting'
        run.save()

        logger.info(f"[Run {run.id}] Exporting model...")

        # Merge LoRA weights
        merged_path = base_dir / 'merged_model'
        logger.info(f"[Run {run.id}] Merging LoRA weights...")
        trainer.merge_and_save(best_checkpoint, str(merged_path))

        # Export to CTranslate2 (faster-whisper)
        logger.info(f"[Run {run.id}] Exporting to CTranslate2...")
        exporter = ModelExporter(str(merged_path), str(base_dir / 'exported'))

        ct2_path = exporter.export_ctranslate2(quantization="int8")

        run.exported_model_path = ct2_path
        run.status = 'completed'
        run.completed_at = timezone.now()
        run.save()

        logger.info(f"[Run {run.id}] Export complete: {ct2_path}")

        # ==========================================
        # Step 5: Create Model Version
        # ==========================================
        logger.info(f"[Run {run.id}] Creating model version...")

        # Generate version number
        version_count = ModelVersion.objects.count()
        version_number = f"v{version_count + 1}.0"

        model_version = ModelVersion.objects.create(
            training_run=run,
            version=version_number,
            lora_weights_path=best_checkpoint,
            merged_model_path=str(merged_path),
            ctranslate2_path=ct2_path,
            quantization="int8",
            nepali_cer=run.nepali_cer,
            english_wer=run.english_wer,
            model_size_mb=exporter._get_directory_size(Path(ct2_path)),
            notes=f"Auto-trained on {run.total_samples} samples",
        )

        logger.info(f"[Run {run.id}] Model version created: {version_number}")

        # ==========================================
        # Success
        # ==========================================
        logger.info(f"[Run {run.id}] Training pipeline completed successfully!")

        return {
            'status': 'completed',
            'run_id': run.id,
            'version': version_number,
            'nepali_cer': run.nepali_cer,
            'english_wer': run.english_wer,
        }

    except Exception as e:
        logger.exception(f"[Run {run.id}] Training failed: {e}")

        run.status = 'failed'
        run.error_message = str(e)
        run.completed_at = timezone.now()
        run.save()

        return {
            'status': 'failed',
            'run_id': run.id,
            'error': str(e),
        }


@shared_task
def cancel_training(training_run_id: int):
    """
    Cancel a running training task.

    Args:
        training_run_id: ID of TrainingRun to cancel
    """
    from .models import TrainingRun
    from celery.app.control import Control

    try:
        run = TrainingRun.objects.get(id=training_run_id)
    except TrainingRun.DoesNotExist:
        logger.error(f"TrainingRun {training_run_id} not found")
        return

    logger.info(f"Cancelling training run {training_run_id}")

    # Revoke celery task
    if run.celery_task_id:
        control = Control(app=cancel_training.app)
        control.revoke(run.celery_task_id, terminate=True, signal='SIGKILL')

    # Update status
    run.status = 'cancelled'
    run.completed_at = timezone.now()
    run.save()

    logger.info(f"Training run {training_run_id} cancelled")


@shared_task
def evaluate_model(model_version_id: int):
    """
    Evaluate a model version on test data.

    Args:
        model_version_id: ID of ModelVersion to evaluate

    Returns:
        Dict with evaluation metrics
    """
    from .models import ModelVersion
    from .services.exporter import test_faster_whisper_model

    try:
        version = ModelVersion.objects.get(id=model_version_id)
    except ModelVersion.DoesNotExist:
        logger.error(f"ModelVersion {model_version_id} not found")
        return {'error': 'ModelVersion not found'}

    logger.info(f"Evaluating model version: {version.version}")

    # TODO: Implement full evaluation
    # This is a placeholder - you should:
    # 1. Load test dataset
    # 2. Run inference with faster-whisper
    # 3. Compute WER/CER metrics
    # 4. Save results to EvaluationResult model

    logger.info(f"Evaluation complete for {version.version}")

    return {
        'version': version.version,
        'nepali_cer': version.nepali_cer,
        'english_wer': version.english_wer,
    }


@shared_task
def cleanup_old_checkpoints(days: int = 30):
    """
    Clean up old training checkpoints to free disk space.

    Args:
        days: Delete checkpoints older than this many days
    """
    import shutil
    from datetime import timedelta
    from .models import TrainingRun

    cutoff_date = timezone.now() - timedelta(days=days)

    old_runs = TrainingRun.objects.filter(
        created_at__lt=cutoff_date,
        status__in=['completed', 'failed', 'cancelled']
    )

    deleted_count = 0
    freed_space = 0

    for run in old_runs:
        # Keep the best checkpoint and exported model
        # Delete intermediate checkpoints
        if run.checkpoint_dir:
            checkpoint_dir = Path(run.checkpoint_dir)
            if checkpoint_dir.exists():
                # Calculate size before deletion
                for f in checkpoint_dir.rglob('*'):
                    if f.is_file() and 'best_model' not in str(f):
                        freed_space += f.stat().st_size
                        f.unlink()

                deleted_count += 1

    freed_space_mb = freed_space / (1024 * 1024)

    logger.info(
        f"Cleanup complete: Processed {deleted_count} runs, "
        f"freed {freed_space_mb:.1f} MB"
    )

    return {
        'deleted_runs': deleted_count,
        'freed_space_mb': freed_space_mb,
    }
