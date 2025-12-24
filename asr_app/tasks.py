"""
Celery tasks for ASR processing
"""

import os
import logging
from celery import shared_task
from celery.signals import worker_process_init
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# Global model cache
_whisper_model = None


@worker_process_init.connect
def preload_whisper_model(**kwargs):
    """Load Whisper model when worker starts"""
    global _whisper_model
    
    import whisper
    
    model_size = getattr(settings, 'WHISPER_MODEL_SIZE', 'small')
    device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
    
    logger.info(f"Preloading Whisper model: {model_size} on {device}")
    _whisper_model = whisper.load_model(model_size, device=device)
    logger.info("Whisper model loaded successfully!")


def get_whisper_model():
    """Get cached model or load if not available"""
    global _whisper_model
    
    if _whisper_model is None:
        import whisper
        model_size = getattr(settings, 'WHISPER_MODEL_SIZE', 'small')
        device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
        _whisper_model = whisper.load_model(model_size, device=device)
    
    return _whisper_model


@shared_task(bind=True, max_retries=3)
def transcribe_audio_task(self, audio_file_id: str, job_id: str):
    """
    Transcribe audio file using local Whisper model
    """
    from .models import AudioFile, TranscriptSegment, TranscriptionJob
    
    try:
        audio_file = AudioFile.objects.get(id=audio_file_id)
        job = TranscriptionJob.objects.get(id=job_id)
        
        job.status = TranscriptionJob.Status.PROCESSING
        job.started_at = timezone.now()
        job.save()
        
        logger.info(f"Starting transcription for {audio_file.original_filename}")
        
        # Get cached model
        model = get_whisper_model()
        
        audio_path = audio_file.file.path
        
        logger.info(f"Transcribing: {audio_path}")
        result = model.transcribe(
            audio_path,
            language=None,
            task='transcribe',
            verbose=False,
            word_timestamps=True,
        )
        
        # Get audio duration
        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        audio_file.duration_seconds = audio_duration
        
        # Create segments
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
        
        audio_file.status = AudioFile.Status.TRANSCRIBED
        audio_file.processed_at = timezone.now()
        audio_file.save()
        
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
        
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))