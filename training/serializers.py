"""
REST API serializers for training models.
"""
from rest_framework import serializers
from .models import TrainingConfig, TrainingRun, ModelVersion, EvaluationResult


class TrainingConfigSerializer(serializers.ModelSerializer):
    """Serializer for TrainingConfig"""

    class Meta:
        model = TrainingConfig
        fields = [
            'id',
            'name',
            'description',
            'base_model',
            'lora_r',
            'lora_alpha',
            'lora_dropout',
            'target_modules',
            'learning_rate',
            'epochs',
            'batch_size',
            'gradient_accumulation_steps',
            'warmup_ratio',
            'weight_decay',
            'max_grad_norm',
            'nepali_ratio',
            'early_stopping_patience',
            'eval_steps',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate_lora_r(self, value):
        """Validate LoRA rank"""
        if value < 1 or value > 256:
            raise serializers.ValidationError("LoRA rank must be between 1 and 256")
        return value

    def validate_learning_rate(self, value):
        """Validate learning rate"""
        if value <= 0 or value > 0.01:
            raise serializers.ValidationError("Learning rate must be between 0 and 0.01")
        return value

    def validate_nepali_ratio(self, value):
        """Validate Nepali ratio"""
        if value < 0 or value > 1:
            raise serializers.ValidationError("Nepali ratio must be between 0 and 1")
        return value


class EvaluationResultSerializer(serializers.ModelSerializer):
    """Serializer for EvaluationResult"""

    class Meta:
        model = EvaluationResult
        fields = [
            'id',
            'test_set',
            'num_samples',
            'wer',
            'cer',
            'insertions',
            'deletions',
            'substitutions',
            'sample_predictions',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class TrainingRunSerializer(serializers.ModelSerializer):
    """Serializer for TrainingRun"""

    config = TrainingConfigSerializer(read_only=True)
    config_id = serializers.PrimaryKeyRelatedField(
        queryset=TrainingConfig.objects.all(),
        source='config',
        write_only=True
    )
    evaluations = EvaluationResultSerializer(many=True, read_only=True)
    duration = serializers.SerializerMethodField()
    improvement = serializers.SerializerMethodField()

    class Meta:
        model = TrainingRun
        fields = [
            'id',
            'config',
            'config_id',
            'status',
            'created_at',
            'started_at',
            'completed_at',
            'duration',
            'total_samples',
            'nepali_samples',
            'english_samples',
            'mixed_samples',
            'total_audio_hours',
            'current_epoch',
            'current_step',
            'total_steps',
            'nepali_cer',
            'nepali_wer',
            'english_wer',
            'english_cer',
            'baseline_nepali_cer',
            'baseline_nepali_wer',
            'baseline_english_wer',
            'baseline_english_cer',
            'improvement',
            'data_dir',
            'checkpoint_dir',
            'best_checkpoint_path',
            'exported_model_path',
            'logs',
            'error_message',
            'celery_task_id',
            'evaluations',
        ]
        read_only_fields = [
            'id',
            'status',
            'created_at',
            'started_at',
            'completed_at',
            'duration',
            'total_samples',
            'nepali_samples',
            'english_samples',
            'mixed_samples',
            'total_audio_hours',
            'current_epoch',
            'current_step',
            'total_steps',
            'nepali_cer',
            'nepali_wer',
            'english_wer',
            'english_cer',
            'baseline_nepali_cer',
            'baseline_nepali_wer',
            'baseline_english_wer',
            'baseline_english_cer',
            'improvement',
            'data_dir',
            'checkpoint_dir',
            'best_checkpoint_path',
            'exported_model_path',
            'logs',
            'error_message',
            'celery_task_id',
            'evaluations',
        ]

    def get_duration(self, obj):
        """Get training duration in seconds"""
        duration = obj.duration
        if duration:
            return duration.total_seconds()
        return None

    def get_improvement(self, obj):
        """Get improvement over baseline"""
        return obj.improvement


class TrainingRunListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing training runs"""

    config_name = serializers.CharField(source='config.name', read_only=True)
    duration = serializers.SerializerMethodField()

    class Meta:
        model = TrainingRun
        fields = [
            'id',
            'config_name',
            'status',
            'created_at',
            'started_at',
            'completed_at',
            'duration',
            'total_samples',
            'nepali_cer',
            'english_wer',
            'error_message',
        ]

    def get_duration(self, obj):
        """Get training duration in seconds"""
        duration = obj.duration
        if duration:
            return duration.total_seconds()
        return None


class ModelVersionSerializer(serializers.ModelSerializer):
    """Serializer for ModelVersion"""

    training_run = TrainingRunListSerializer(read_only=True)
    training_run_id = serializers.PrimaryKeyRelatedField(
        queryset=TrainingRun.objects.all(),
        source='training_run',
        write_only=True
    )

    class Meta:
        model = ModelVersion
        fields = [
            'id',
            'training_run',
            'training_run_id',
            'version',
            'is_active',
            'lora_weights_path',
            'merged_model_path',
            'ctranslate2_path',
            'ggml_path',
            'quantization',
            'model_size_mb',
            'nepali_cer',
            'english_wer',
            'avg_rtf',
            'created_at',
            'notes',
        ]
        read_only_fields = [
            'id',
            'training_run',
            'created_at',
        ]

    def validate_version(self, value):
        """Validate version format"""
        import re
        if not re.match(r'^v\d+\.\d+$', value):
            raise serializers.ValidationError(
                "Version must be in format 'vX.Y' (e.g., 'v1.0')"
            )
        return value


class ModelVersionListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing model versions"""

    class Meta:
        model = ModelVersion
        fields = [
            'id',
            'version',
            'is_active',
            'quantization',
            'model_size_mb',
            'nepali_cer',
            'english_wer',
            'avg_rtf',
            'created_at',
        ]
