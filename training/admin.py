"""
Django admin interface for training models.
"""
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import TrainingConfig, TrainingRun, ModelVersion, EvaluationResult


@admin.register(TrainingConfig)
class TrainingConfigAdmin(admin.ModelAdmin):
    """Admin interface for TrainingConfig"""

    list_display = [
        'name',
        'base_model',
        'lora_r',
        'lora_alpha',
        'learning_rate',
        'epochs',
        'batch_size',
        'created_at',
    ]
    list_filter = ['base_model', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']

    fieldsets = [
        ('Basic Info', {
            'fields': ['name', 'description', 'base_model']
        }),
        ('LoRA Settings', {
            'fields': [
                'lora_r',
                'lora_alpha',
                'lora_dropout',
                'target_modules',
            ],
            'description': 'LoRA (Low-Rank Adaptation) hyperparameters'
        }),
        ('Training Hyperparameters', {
            'fields': [
                'learning_rate',
                'epochs',
                'batch_size',
                'gradient_accumulation_steps',
                'warmup_ratio',
                'weight_decay',
                'max_grad_norm',
            ]
        }),
        ('Language & Evaluation', {
            'fields': [
                'nepali_ratio',
                'early_stopping_patience',
                'eval_steps',
            ]
        }),
        ('Metadata', {
            'fields': ['created_at', 'updated_at'],
            'classes': ['collapse']
        }),
    ]

    def get_readonly_fields(self, request, obj=None):
        """Make created_at and updated_at readonly"""
        return ['created_at', 'updated_at']


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    """Admin interface for TrainingRun"""

    list_display = [
        'id',
        'config_name',
        'status_badge',
        'total_samples',
        'nepali_cer_display',
        'english_wer_display',
        'duration_display',
        'created_at',
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['id', 'config__name']
    readonly_fields = [
        'created_at',
        'started_at',
        'completed_at',
        'duration_display',
        'improvement_display',
        'celery_task_id',
    ]

    fieldsets = [
        ('Configuration', {
            'fields': ['config']
        }),
        ('Status', {
            'fields': [
                'status',
                'created_at',
                'started_at',
                'completed_at',
                'duration_display',
                'celery_task_id',
            ]
        }),
        ('Data Statistics', {
            'fields': [
                'total_samples',
                'nepali_samples',
                'english_samples',
                'mixed_samples',
                'total_audio_hours',
            ]
        }),
        ('Training Progress', {
            'fields': [
                'current_epoch',
                'current_step',
                'total_steps',
            ]
        }),
        ('Metrics', {
            'fields': [
                'nepali_cer',
                'nepali_wer',
                'english_wer',
                'english_cer',
                'baseline_nepali_cer',
                'baseline_nepali_wer',
                'baseline_english_wer',
                'baseline_english_cer',
                'improvement_display',
            ]
        }),
        ('Paths', {
            'fields': [
                'data_dir',
                'checkpoint_dir',
                'best_checkpoint_path',
                'exported_model_path',
            ],
            'classes': ['collapse']
        }),
        ('Logs', {
            'fields': ['logs', 'error_message'],
            'classes': ['collapse']
        }),
    ]

    def config_name(self, obj):
        """Display config name"""
        return obj.config.name
    config_name.short_description = 'Config'

    def status_badge(self, obj):
        """Display status as colored badge"""
        colors = {
            'pending': 'gray',
            'preparing': 'blue',
            'training': 'orange',
            'evaluating': 'purple',
            'exporting': 'cyan',
            'completed': 'green',
            'failed': 'red',
            'cancelled': 'darkgray',
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 10px; border-radius: 3px;">{}</span>',
            color,
            obj.status.upper()
        )
    status_badge.short_description = 'Status'

    def nepali_cer_display(self, obj):
        """Display Nepali CER as percentage"""
        if obj.nepali_cer is not None:
            return f'{obj.nepali_cer:.2%}'
        return '-'
    nepali_cer_display.short_description = 'Nepali CER'

    def english_wer_display(self, obj):
        """Display English WER as percentage"""
        if obj.english_wer is not None:
            return f'{obj.english_wer:.2%}'
        return '-'
    english_wer_display.short_description = 'English WER'

    def duration_display(self, obj):
        """Display training duration"""
        duration = obj.duration
        if duration:
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f'{hours}h {minutes}m {seconds}s'
        return '-'
    duration_display.short_description = 'Duration'

    def improvement_display(self, obj):
        """Display improvement over baseline"""
        improvement = obj.improvement
        if improvement:
            lines = []
            if 'nepali_cer_pct' in improvement:
                lines.append(f"Nepali CER: {improvement['nepali_cer_pct']:+.1f}%")
            if 'english_wer_pct' in improvement:
                lines.append(f"English WER: {improvement['english_wer_pct']:+.1f}%")
            return mark_safe('<br>'.join(lines))
        return '-'
    improvement_display.short_description = 'Improvement'


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    """Admin interface for ModelVersion"""

    list_display = [
        'version',
        'is_active_badge',
        'quantization',
        'model_size_mb',
        'nepali_cer_display',
        'english_wer_display',
        'avg_rtf',
        'created_at',
    ]
    list_filter = ['is_active', 'quantization', 'created_at']
    search_fields = ['version', 'notes']
    readonly_fields = ['created_at']

    fieldsets = [
        ('Version Info', {
            'fields': ['version', 'is_active', 'training_run']
        }),
        ('Model Paths', {
            'fields': [
                'lora_weights_path',
                'merged_model_path',
                'ctranslate2_path',
                'ggml_path',
            ]
        }),
        ('Model Info', {
            'fields': [
                'quantization',
                'model_size_mb',
            ]
        }),
        ('Performance Metrics', {
            'fields': [
                'nepali_cer',
                'english_wer',
                'avg_rtf',
            ]
        }),
        ('Metadata', {
            'fields': ['created_at', 'notes']
        }),
    ]

    def is_active_badge(self, obj):
        """Display is_active as badge"""
        if obj.is_active:
            return format_html(
                '<span style="background-color: green; color: white; '
                'padding: 3px 10px; border-radius: 3px;">ACTIVE</span>'
            )
        return format_html(
            '<span style="color: gray;">Inactive</span>'
        )
    is_active_badge.short_description = 'Status'

    def nepali_cer_display(self, obj):
        """Display Nepali CER as percentage"""
        return f'{obj.nepali_cer:.2%}'
    nepali_cer_display.short_description = 'Nepali CER'

    def english_wer_display(self, obj):
        """Display English WER as percentage"""
        return f'{obj.english_wer:.2%}'
    english_wer_display.short_description = 'English WER'


@admin.register(EvaluationResult)
class EvaluationResultAdmin(admin.ModelAdmin):
    """Admin interface for EvaluationResult"""

    list_display = [
        'training_run',
        'test_set',
        'num_samples',
        'wer_display',
        'cer_display',
        'created_at',
    ]
    list_filter = ['test_set', 'created_at']
    search_fields = ['training_run__id']
    readonly_fields = ['created_at']

    fieldsets = [
        ('Test Info', {
            'fields': ['training_run', 'test_set', 'num_samples']
        }),
        ('Metrics', {
            'fields': ['wer', 'cer']
        }),
        ('Error Breakdown', {
            'fields': ['insertions', 'deletions', 'substitutions']
        }),
        ('Sample Predictions', {
            'fields': ['sample_predictions'],
            'classes': ['collapse']
        }),
        ('Metadata', {
            'fields': ['created_at']
        }),
    ]

    def wer_display(self, obj):
        """Display WER as percentage"""
        return f'{obj.wer:.2%}'
    wer_display.short_description = 'WER'

    def cer_display(self, obj):
        """Display CER as percentage"""
        return f'{obj.cer:.2%}'
    cer_display.short_description = 'CER'
