"""
Serializers for ASR API
"""

from rest_framework import serializers
from .models import AudioFile, TranscriptSegment, Correction, TranscriptionJob


class TranscriptSegmentSerializer(serializers.ModelSerializer):
    """Serializer for transcript segments"""
    
    final_text = serializers.ReadOnlyField()
    duration = serializers.ReadOnlyField()
    
    class Meta:
        model = TranscriptSegment
        fields = [
            'id', 'start_time', 'end_time', 'duration',
            'text', 'corrected_text', 'final_text',
            'confidence', 'language', 'is_corrected',
            'corrected_at', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class SegmentCorrectionSerializer(serializers.Serializer):
    """Serializer for submitting corrections"""
    
    corrected_text = serializers.CharField(required=True)
    correction_type = serializers.ChoiceField(
        choices=Correction.CorrectionType.choices,
        default=Correction.CorrectionType.WORD
    )
    notes = serializers.CharField(required=False, allow_blank=True)


class AudioFileListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing audio files"""
    
    segment_count = serializers.SerializerMethodField()
    corrected_count = serializers.SerializerMethodField()
    
    class Meta:
        model = AudioFile
        fields = [
            'id', 'original_filename', 'title', 'status',
            'language', 'duration_seconds', 'segment_count',
            'corrected_count', 'created_at', 'updated_at'
        ]
    
    def get_segment_count(self, obj):
        return obj.segments.count()
    
    def get_corrected_count(self, obj):
        return obj.segments.filter(is_corrected=True).count()


class AudioFileDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer with segments"""
    
    segments = TranscriptSegmentSerializer(many=True, read_only=True)
    full_transcript = serializers.ReadOnlyField()
    file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = AudioFile
        fields = [
            'id', 'file', 'file_url', 'original_filename', 'title',
            'description', 'file_size', 'duration_seconds', 
            'sample_rate', 'status', 'language', 'error_message',
            'tags', 'segments', 'full_transcript',
            'created_at', 'updated_at', 'processed_at'
        ]
        read_only_fields = [
            'id', 'file_size', 'duration_seconds', 'sample_rate',
            'status', 'error_message', 'created_at', 'updated_at', 
            'processed_at'
        ]
    
    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and request:
            return request.build_absolute_uri(obj.file.url)
        return None


class AudioFileUploadSerializer(serializers.ModelSerializer):
    """Serializer for uploading audio files"""
    
    class Meta:
        model = AudioFile
        fields = ['file', 'title', 'description', 'language', 'tags']
    
    def validate_file(self, value):
        # Validate file size (max 100MB)
        max_size = 100 * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File size ({value.size / 1024 / 1024:.1f}MB) exceeds maximum allowed (100MB)"
            )
        
        # Validate file type
        allowed_types = [
            'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave',
            'audio/x-wav', 'audio/ogg', 'audio/flac', 'audio/m4a',
            'audio/mp4', 'audio/webm', 'video/webm'
        ]
        content_type = value.content_type
        if content_type not in allowed_types:
            raise serializers.ValidationError(
                f"Unsupported audio format: {content_type}"
            )
        
        return value
    
    def create(self, validated_data):
        file = validated_data['file']
        validated_data['original_filename'] = file.name
        validated_data['file_size'] = file.size
        return super().create(validated_data)


class TranscriptionJobSerializer(serializers.ModelSerializer):
    """Serializer for transcription jobs"""
    
    class Meta:
        model = TranscriptionJob
        fields = [
            'id', 'audio_file', 'celery_task_id', 'status',
            'progress', 'result', 'error_message',
            'started_at', 'completed_at', 'created_at'
        ]
        read_only_fields = fields


class CorrectionSerializer(serializers.ModelSerializer):
    """Serializer for correction history"""
    
    class Meta:
        model = Correction
        fields = [
            'id', 'segment', 'original_text', 'corrected_text',
            'correction_type', 'notes', 'is_exported',
            'created_at'
        ]
        read_only_fields = ['id', 'is_exported', 'created_at']


class BulkCorrectionSerializer(serializers.Serializer):
    """Serializer for bulk corrections"""
    
    corrections = serializers.ListField(
        child=serializers.DictField(),
        min_length=1
    )
    
    def validate_corrections(self, value):
        for item in value:
            if 'segment_id' not in item:
                raise serializers.ValidationError("Each correction must have segment_id")
            if 'corrected_text' not in item:
                raise serializers.ValidationError("Each correction must have corrected_text")
        return value


class ExportDataSerializer(serializers.Serializer):
    """Serializer for training data export"""
    
    format = serializers.ChoiceField(
        choices=['json', 'csv', 'huggingface'],
        default='json'
    )
    include_uncorrected = serializers.BooleanField(default=False)
    language_filter = serializers.ChoiceField(
        choices=[('all', 'All'), ('ne', 'Nepali'), ('en', 'English'), ('mixed', 'Mixed')],
        default='all'
    )


# ===================
# Audio Chat Serializers
# ===================

class AudioChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for audio chat messages"""
    
    class Meta:
        from .models import AudioChatMessage
        model = AudioChatMessage
        fields = [
            'id', 'role', 'status',
            'original_text', 'translated_text',
            'audio_file', 'translated_audio',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'status', 'original_text', 'translated_text',
            'translated_audio', 'created_at', 'updated_at'
        ]


class AudioChatSerializer(serializers.ModelSerializer):
    """Serializer for audio chat"""
    
    messages = AudioChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        from .models import AudioChat
        model = AudioChat
        fields = [
            'id', 'title', 'status',
            'source_language', 'target_language',
            'auto_play_translation',
            'messages', 'message_count',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        return obj.messages.count()


class AudioChatCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating audio chat"""
    
    class Meta:
        from .models import AudioChat
        model = AudioChat
        fields = ['id', 'title', 'source_language', 'target_language', 'auto_play_translation', 'created_at']
        read_only_fields = ['id', 'created_at']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set defaults
        self.fields['source_language'].default = 'mixed'
        self.fields['target_language'].default = 'en'
        self.fields['auto_play_translation'].default = True
        self.fields['title'].required = False
        self.fields['title'].allow_blank = True
