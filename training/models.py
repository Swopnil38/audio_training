"""
Training models for Whisper LoRA fine-tuning

Models:
- TrainingConfig: Reusable training configuration (hyperparameters, LoRA settings)
- TrainingRun: Individual training run with status tracking
- ModelVersion: Deployed model versions for A/B testing and rollback
- EvaluationResult: Detailed evaluation results per test set
"""

from django.db import models
from django.utils import timezone


class TrainingConfig(models.Model):
    """Reusable training configuration"""

    name = models.CharField(max_length=100, help_text="Configuration name")
    description = models.TextField(blank=True, help_text="Configuration description")

    # Model settings
    base_model = models.CharField(
        max_length=100,
        default="openai/whisper-small",
        help_text="HuggingFace model ID"
    )

    # LoRA settings
    lora_r = models.IntegerField(
        default=16,
        help_text="LoRA rank (lower=fewer params, typically 8-32)"
    )
    lora_alpha = models.IntegerField(
        default=32,
        help_text="LoRA alpha (usually 2x rank)"
    )
    lora_dropout = models.FloatField(
        default=0.1,
        help_text="LoRA dropout rate"
    )
    target_modules = models.JSONField(
        default=list,
        help_text="Modules to apply LoRA (e.g., ['q_proj', 'v_proj'])"
    )

    # Training hyperparameters
    learning_rate = models.FloatField(
        default=1e-4,
        help_text="Learning rate (typically 1e-4 to 1e-5)"
    )
    epochs = models.IntegerField(
        default=10,
        help_text="Number of training epochs"
    )
    batch_size = models.IntegerField(
        default=8,
        help_text="Per-device batch size"
    )
    gradient_accumulation_steps = models.IntegerField(
        default=4,
        help_text="Gradient accumulation steps (effective batch = batch_size * this)"
    )
    warmup_ratio = models.FloatField(
        default=0.1,
        help_text="Warmup ratio (0.0 to 0.2)"
    )
    weight_decay = models.FloatField(
        default=0.01,
        help_text="Weight decay for regularization"
    )
    max_grad_norm = models.FloatField(
        default=1.0,
        help_text="Maximum gradient norm for clipping"
    )

    # Language balancing
    nepali_ratio = models.FloatField(
        default=0.6,
        help_text="Target ratio of Nepali samples in training data (0-1)"
    )

    # Early stopping
    early_stopping_patience = models.IntegerField(
        default=5,
        help_text="Patience for early stopping"
    )
    eval_steps = models.IntegerField(
        default=500,
        help_text="Evaluate every N steps"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} (r={self.lora_r}, lr={self.learning_rate})"

    def save(self, *args, **kwargs):
        """Set default target modules if empty"""
        if not self.target_modules:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
        super().save(*args, **kwargs)


class TrainingRun(models.Model):
    """Individual training run with status tracking"""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('preparing', 'Preparing Data'),
        ('training', 'Training'),
        ('evaluating', 'Evaluating'),
        ('exporting', 'Exporting Model'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]

    config = models.ForeignKey(
        TrainingConfig,
        on_delete=models.CASCADE,
        related_name='runs'
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        db_index=True
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Data statistics
    total_samples = models.IntegerField(default=0)
    nepali_samples = models.IntegerField(default=0)
    english_samples = models.IntegerField(default=0)
    mixed_samples = models.IntegerField(default=0)
    total_audio_hours = models.FloatField(default=0)

    # Training progress
    current_epoch = models.IntegerField(default=0)
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)

    # Final metrics (after evaluation)
    nepali_cer = models.FloatField(
        null=True,
        blank=True,
        help_text="Character Error Rate for Nepali (Devanagari)"
    )
    nepali_wer = models.FloatField(
        null=True,
        blank=True,
        help_text="Word Error Rate for Nepali"
    )
    english_wer = models.FloatField(
        null=True,
        blank=True,
        help_text="Word Error Rate for English"
    )
    english_cer = models.FloatField(
        null=True,
        blank=True,
        help_text="Character Error Rate for English"
    )

    # Baseline metrics (before training, for comparison)
    baseline_nepali_cer = models.FloatField(null=True, blank=True)
    baseline_nepali_wer = models.FloatField(null=True, blank=True)
    baseline_english_wer = models.FloatField(null=True, blank=True)
    baseline_english_cer = models.FloatField(null=True, blank=True)

    # Paths
    data_dir = models.CharField(max_length=500, blank=True)
    checkpoint_dir = models.CharField(max_length=500, blank=True)
    best_checkpoint_path = models.CharField(max_length=500, blank=True)
    exported_model_path = models.CharField(max_length=500, blank=True)

    # Logs and errors
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)

    # Celery task ID for cancellation
    celery_task_id = models.CharField(max_length=100, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return f"Run #{self.id} - {self.config.name} ({self.status})"

    @property
    def duration(self):
        """Calculate training duration"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def improvement(self):
        """Calculate improvement over baseline"""
        improvements = {}

        if self.nepali_cer and self.baseline_nepali_cer:
            diff = self.baseline_nepali_cer - self.nepali_cer
            improvements['nepali_cer'] = diff
            improvements['nepali_cer_pct'] = (diff / self.baseline_nepali_cer) * 100

        if self.english_wer and self.baseline_english_wer:
            diff = self.baseline_english_wer - self.english_wer
            improvements['english_wer'] = diff
            improvements['english_wer_pct'] = (diff / self.baseline_english_wer) * 100

        return improvements if improvements else None


class ModelVersion(models.Model):
    """Deployed model versions for A/B testing and rollback"""

    training_run = models.OneToOneField(
        TrainingRun,
        on_delete=models.CASCADE,
        related_name='model_version'
    )

    version = models.CharField(
        max_length=50,
        unique=True,
        help_text="Version string (e.g., 'v1.0', 'v1.1')"
    )
    is_active = models.BooleanField(
        default=False,
        help_text="Currently deployed model"
    )

    # Model paths
    lora_weights_path = models.CharField(
        max_length=500,
        help_text="Path to LoRA adapter weights"
    )
    merged_model_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to merged full model"
    )
    ctranslate2_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to CTranslate2/faster-whisper format"
    )
    ggml_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to GGML/whisper.cpp format"
    )

    # Quantization info
    quantization = models.CharField(
        max_length=20,
        default="int8",
        help_text="Quantization type (int8, int4, float16, float32)"
    )
    model_size_mb = models.IntegerField(
        null=True,
        blank=True,
        help_text="Model size in megabytes"
    )

    # Performance metrics
    nepali_cer = models.FloatField(help_text="Nepali Character Error Rate")
    english_wer = models.FloatField(help_text="English Word Error Rate")
    avg_rtf = models.FloatField(
        null=True,
        blank=True,
        help_text="Average real-time factor (lower=faster, 1.0=real-time)"
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, help_text="Deployment notes")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        active = " (ACTIVE)" if self.is_active else ""
        return f"{self.version}{active} - CER: {self.nepali_cer:.2%}, WER: {self.english_wer:.2%}"

    def save(self, *args, **kwargs):
        """Ensure only one active model at a time"""
        if self.is_active:
            # Deactivate all other models
            ModelVersion.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)


class EvaluationResult(models.Model):
    """Detailed evaluation results per test set"""

    TEST_SET_CHOICES = [
        ('nepali_clean', 'Nepali Clean'),
        ('nepali_noisy', 'Nepali Noisy'),
        ('english', 'English'),
        ('mixed', 'Code-switched'),
    ]

    training_run = models.ForeignKey(
        TrainingRun,
        on_delete=models.CASCADE,
        related_name='evaluations'
    )

    test_set = models.CharField(
        max_length=20,
        choices=TEST_SET_CHOICES
    )
    num_samples = models.IntegerField(help_text="Number of test samples")

    # Metrics
    wer = models.FloatField(help_text="Word Error Rate")
    cer = models.FloatField(help_text="Character Error Rate")

    # Detailed error breakdown
    insertions = models.IntegerField(default=0)
    deletions = models.IntegerField(default=0)
    substitutions = models.IntegerField(default=0)

    # Sample predictions for debugging
    sample_predictions = models.JSONField(
        default=list,
        help_text="Sample predictions: [{ref, hyp, wer, cer}, ...]"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['training_run', 'test_set']
        ordering = ['training_run', 'test_set']

    def __str__(self):
        return f"{self.training_run} - {self.test_set}: WER={self.wer:.2%}, CER={self.cer:.2%}"
