"""
Admin configuration for ASR System
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import AudioFile, TranscriptSegment, Correction, TranscriptionJob, AudioChat, AudioChatMessage


@admin.register(AudioFile)
class AudioFileAdmin(admin.ModelAdmin):
    list_display = [
        'original_filename', 'status', 'language', 
        'duration_display', 'segment_count', 'created_at'
    ]
    list_filter = ['status', 'language', 'created_at']
    search_fields = ['original_filename', 'title', 'description']
    readonly_fields = [
        'id', 'file_size', 'duration_seconds', 'sample_rate',
        'created_at', 'updated_at', 'processed_at'
    ]
    
    def duration_display(self, obj):
        if obj.duration_seconds:
            mins = int(obj.duration_seconds // 60)
            secs = int(obj.duration_seconds % 60)
            return f"{mins}:{secs:02d}"
        return "-"
    duration_display.short_description = "Duration"
    
    def segment_count(self, obj):
        return obj.segments.count()
    segment_count.short_description = "Segments"


@admin.register(TranscriptSegment)
class TranscriptSegmentAdmin(admin.ModelAdmin):
    list_display = [
        'audio_file', 'time_range', 'text_preview', 
        'is_corrected', 'created_at'
    ]
    list_filter = ['is_corrected', 'language', 'audio_file']
    search_fields = ['text', 'corrected_text']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    def time_range(self, obj):
        return f"{obj.start_time:.2f}s - {obj.end_time:.2f}s"
    time_range.short_description = "Time"
    
    def text_preview(self, obj):
        text = obj.final_text
        return text[:80] + "..." if len(text) > 80 else text
    text_preview.short_description = "Text"


@admin.register(Correction)
class CorrectionAdmin(admin.ModelAdmin):
    list_display = [
        'segment', 'correction_type', 'original_preview',
        'corrected_preview', 'is_exported', 'created_at'
    ]
    list_filter = ['correction_type', 'is_exported', 'created_at']
    search_fields = ['original_text', 'corrected_text', 'notes']
    readonly_fields = ['id', 'created_at']
    
    def original_preview(self, obj):
        return obj.original_text[:50] + "..." if len(obj.original_text) > 50 else obj.original_text
    original_preview.short_description = "Original"
    
    def corrected_preview(self, obj):
        return obj.corrected_text[:50] + "..." if len(obj.corrected_text) > 50 else obj.corrected_text
    corrected_preview.short_description = "Corrected"


@admin.register(TranscriptionJob)
class TranscriptionJobAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'audio_file', 'status', 'progress',
        'started_at', 'completed_at'
    ]
    list_filter = ['status', 'created_at']
    readonly_fields = [
        'id', 'celery_task_id', 'result', 'error_message',
        'started_at', 'completed_at', 'created_at'
    ]


# ===================
# Audio Chat Admin
# ===================

@admin.register(AudioChat)
class AudioChatAdmin(admin.ModelAdmin):
    from .models import AudioChat
    list_display = [
        'title', 'user', 'status', 'source_language', 'target_language',
        'message_count', 'created_at', 'updated_at'
    ]
    list_filter = ['status', 'source_language', 'created_at']
    search_fields = ['title', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = "Messages"


@admin.register(AudioChatMessage)
class AudioChatMessageAdmin(admin.ModelAdmin):
    from .models import AudioChatMessage
    list_display = [
        'id', 'chat', 'role', 'status', 'original_text_preview', 'created_at'
    ]
    list_filter = ['status', 'role', 'created_at']
    search_fields = ['original_text', 'translated_text']
    readonly_fields = [
        'id', 'original_text', 'translated_text', 'processing_metadata',
        'created_at', 'updated_at'
    ]
    
    def original_text_preview(self, obj):
        if obj.original_text:
            return obj.original_text[:50] + ('...' if len(obj.original_text) > 50 else '')
        return '-'
    original_text_preview.short_description = "Original Text"

