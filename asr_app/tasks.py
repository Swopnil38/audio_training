"""
Celery tasks for ASR processing
"""

import os
import tempfile
import logging
from celery import shared_task
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def transcribe_audio_task(self, audio_file_id: str, job_id: str):
    """
    Transcribe audio file using local Whisper model
    """
    from .models import AudioFile, TranscriptSegment, TranscriptionJob
    
    try:
        # Get objects
        audio_file = AudioFile.objects.get(id=audio_file_id)
        job = TranscriptionJob.objects.get(id=job_id)
        
        # Update job status
        job.status = TranscriptionJob.Status.PROCESSING
        job.started_at = timezone.now()
        job.save()
        
        logger.info(f"Starting transcription for {audio_file.original_filename}")
        
        # Import Whisper (lazy load to avoid GPU memory issues)
        import whisper
        import numpy as np
        
        # Load model
        model_size = getattr(settings, 'WHISPER_MODEL_SIZE', 'small')
        device = getattr(settings, 'WHISPER_DEVICE', 'cuda')
        
        logger.info(f"Loading Whisper model: {model_size} on {device}")
        model = whisper.load_model(model_size, device=device)
        
        # Get audio file path
        audio_path = audio_file.file.path
        
        # Transcribe with word timestamps
        logger.info(f"Transcribing: {audio_path}")
        result = model.transcribe(
            audio_path,
            language=None,  # Auto-detect (for Nepali + English)
            task='transcribe',
            verbose=False,
            word_timestamps=True,
        )
        
        # Get audio duration
        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        audio_file.duration_seconds = audio_duration
        
        # Create segments from result
        segments_created = 0
        for segment in result.get('segments', []):
            TranscriptSegment.objects.create(
                audio_file=audio_file,
                start_time=segment['start'],
                end_time=segment['end'],
                text=segment['text'].strip(),
                confidence=segment.get('avg_logprob'),
                no_speech_prob=segment.get('no_speech_prob'),
                language=result.get('language', ''),
            )
            segments_created += 1
        
        # Update audio file
        audio_file.status = AudioFile.Status.TRANSCRIBED
        audio_file.processed_at = timezone.now()
        audio_file.save()
        
        # Update job
        job.status = TranscriptionJob.Status.COMPLETED
        job.progress = 100
        job.completed_at = timezone.now()
        job.result = {
            'language': result.get('language'),
            'segments_count': segments_created,
            'duration': audio_duration,
        }
        job.save()
        
        logger.info(f"Transcription complete: {segments_created} segments created")
        
        return {
            'status': 'success',
            'segments': segments_created,
            'language': result.get('language')
        }
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        
        # Update job with error
        try:
            job = TranscriptionJob.objects.get(id=job_id)
            job.status = TranscriptionJob.Status.FAILED
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()
            
            audio_file = AudioFile.objects.get(id=audio_file_id)
            audio_file.status = AudioFile.Status.FAILED
            audio_file.error_message = str(e)
            audio_file.save()
        except Exception:
            pass
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@shared_task
def generate_embeddings_task(segment_id: str):
    """
    Generate embeddings for a transcript segment
    """
    from .models import TranscriptSegment
    from sentence_transformers import SentenceTransformer
    
    try:
        segment = TranscriptSegment.objects.get(id=segment_id)
        
        # Use multilingual model that works well with Nepali
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Get text to embed
        text = segment.final_text
        
        # Generate embedding
        embedding = model.encode(text)
        
        # Save to segment
        segment.embedding = embedding.tolist()
        segment.save()
        
        logger.info(f"Generated embedding for segment {segment_id}")
        
        return {'status': 'success', 'segment_id': str(segment_id)}
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def batch_generate_embeddings_task(audio_file_id: str):
    """
    Generate embeddings for all segments in an audio file
    """
    from .models import AudioFile, TranscriptSegment
    from sentence_transformers import SentenceTransformer
    
    try:
        audio_file = AudioFile.objects.get(id=audio_file_id)
        segments = audio_file.segments.all()
        
        if not segments.exists():
            return {'status': 'no_segments'}
        
        # Load model once
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Get all texts
        texts = [seg.final_text for seg in segments]
        
        # Generate embeddings in batch
        embeddings = model.encode(texts)
        
        # Save embeddings
        for segment, embedding in zip(segments, embeddings):
            segment.embedding = embedding.tolist()
            segment.save()
        
        logger.info(f"Generated {len(embeddings)} embeddings for audio {audio_file_id}")
        
        return {'status': 'success', 'count': len(embeddings)}
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def cleanup_old_jobs():
    """
    Cleanup old completed/failed jobs
    """
    from .models import TranscriptionJob
    from datetime import timedelta
    
    cutoff = timezone.now() - timedelta(days=7)
    
    deleted, _ = TranscriptionJob.objects.filter(
        status__in=[TranscriptionJob.Status.COMPLETED, TranscriptionJob.Status.FAILED],
        created_at__lt=cutoff
    ).delete()
    
    logger.info(f"Cleaned up {deleted} old jobs")
    return {'deleted': deleted}
