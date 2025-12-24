"""
Models for ASR System
- AudioFile: uploaded audio metadata
- TranscriptSegment: timestamped transcription segments
- Correction: human corrections for training
"""

import uuid
from django.db import models
from django.contrib.auth import get_user_model
from pgvector.django import VectorField

User = get_user_model()


class AudioFile(models.Model):
    """Audio file metadata and processing status"""
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        TRANSCRIBED = 'transcribed', 'Transcribed'
        CORRECTED = 'corrected', 'Corrected'
        APPROVED = 'approved', 'Approved'
        FAILED = 'failed', 'Failed'
    
    class Language(models.TextChoices):
        NEPALI = 'ne', 'Nepali'
        ENGLISH = 'en', 'English'
        MIXED = 'mixed', 'Mixed (Nepali + English)'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # File info
    file = models.FileField(upload_to='audio/%Y/%m/%d/')
    original_filename = models.CharField(max_length=255)
    file_size = models.PositiveIntegerField(help_text='File size in bytes')
    duration_seconds = models.FloatField(null=True, blank=True)
    sample_rate = models.PositiveIntegerField(null=True, blank=True)
    
    # Processing
    status = models.CharField(
        max_length=20, 
        choices=Status.choices, 
        default=Status.PENDING,
        db_index=True
    )
    language = models.CharField(
        max_length=10, 
        choices=Language.choices, 
        default=Language.MIXED
    )
    error_message = models.TextField(blank=True)
    
    # Metadata
    title = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    tags = models.JSONField(default=list, blank=True)
    
    # Tracking
    uploaded_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='audio_files'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.original_filename} ({self.status})"
    
    @property
    def full_transcript(self):
        """Get complete transcript text"""
        segments = self.segments.order_by('start_time')
        return ' '.join(seg.text for seg in segments)


class TranscriptSegment(models.Model):
    """Individual transcript segment with timestamps"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    audio_file = models.ForeignKey(
        AudioFile, 
        on_delete=models.CASCADE, 
        related_name='segments'
    )
    
    # Timing
    start_time = models.FloatField(help_text='Start time in seconds')
    end_time = models.FloatField(help_text='End time in seconds')
    
    # Transcription
    text = models.TextField(help_text='Original Whisper transcription')
    corrected_text = models.TextField(
        blank=True, 
        help_text='Human-corrected text'
    )
    
    # Whisper metadata
    confidence = models.FloatField(null=True, blank=True)
    language = models.CharField(max_length=10, blank=True)
    no_speech_prob = models.FloatField(null=True, blank=True)
    
    # Embedding for semantic search
    embedding = VectorField(dimensions=384, null=True, blank=True)
    
    # Status
    is_corrected = models.BooleanField(default=False)
    corrected_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='corrected_segments'
    )
    corrected_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['audio_file', 'start_time']
        indexes = [
            models.Index(fields=['audio_file', 'start_time']),
            models.Index(fields=['is_corrected']),
        ]
    
    def __str__(self):
        return f"[{self.start_time:.2f}-{self.end_time:.2f}] {self.text[:50]}..."
    
    @property
    def duration(self):
        return self.end_time - self.start_time
    
    @property
    def final_text(self):
        """Return corrected text if available, else original"""
        return self.corrected_text if self.corrected_text else self.text


class Correction(models.Model):
    """
    Tracks correction history for training data export.
    Each correction creates a training pair: (audio_chunk, corrected_text)
    """
    
    class CorrectionType(models.TextChoices):
        TYPO = 'typo', 'Typo Fix'
        WORD = 'word', 'Word Correction'
        PHRASE = 'phrase', 'Phrase Rewrite'
        PUNCTUATION = 'punctuation', 'Punctuation'
        LANGUAGE = 'language', 'Language Tag'
        OTHER = 'other', 'Other'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    segment = models.ForeignKey(
        TranscriptSegment,
        on_delete=models.CASCADE,
        related_name='corrections'
    )
    
    # Correction details
    original_text = models.TextField()
    corrected_text = models.TextField()
    correction_type = models.CharField(
        max_length=20,
        choices=CorrectionType.choices,
        default=CorrectionType.WORD
    )
    notes = models.TextField(blank=True, help_text='Correction notes')
    
    # Training data export
    is_exported = models.BooleanField(default=False)
    exported_at = models.DateTimeField(null=True, blank=True)
    
    # Tracking
    corrected_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Correction: '{self.original_text[:30]}' → '{self.corrected_text[:30]}'"


class TranscriptionJob(models.Model):
    """Track async transcription jobs"""
    
    class Status(models.TextChoices):
        QUEUED = 'queued', 'Queued'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    audio_file = models.ForeignKey(
        AudioFile,
        on_delete=models.CASCADE,
        related_name='jobs'
    )
    
    celery_task_id = models.CharField(max_length=255, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.QUEUED
    )
    progress = models.PositiveIntegerField(default=0)  # 0-100
    
    # Results
    result = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    # Timing
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Job {self.id} - {self.status}"
