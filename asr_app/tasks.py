"""
Celery tasks for ASR processing - Using Fine-tuned Nepali Whisper
"""

import os
import logging
import librosa
import requests
from celery import shared_task
from celery.signals import worker_process_init
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)





def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Hugging Face wav2vec2-xlsr-nepali model"""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch
    import os
    # Path to local model directory (adjust if needed)
    LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'wav2vec2-xlsr-nepali')
    processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL_DIR)

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    duration = len(audio) / sr
    return {
        'segments': [{
            'start': 0,
            'end': duration,
            'text': transcription
        }],
        'language': 'ne',
        'duration': duration
    }


@shared_task(bind=True, max_retries=3)
def transcribe_audio_task(self, audio_file_id: str, job_id: str):
    """
    Transcribe audio file using fine-tuned Nepali Whisper model
    """
    from .models import AudioFile, TranscriptSegment, TranscriptionJob
    
    try:
        audio_file = AudioFile.objects.get(id=audio_file_id)
        job = TranscriptionJob.objects.get(id=job_id)
        
        job.status = TranscriptionJob.Status.PROCESSING
        job.started_at = timezone.now()
        job.save()
        
        logger.info(f"Starting transcription for {audio_file.original_filename}")
        
        audio_path = audio_file.file.path
        logger.info(f"Transcribing: {audio_path}")
        
        # Transcribe
        result = transcribe_audio(audio_path)
        
        # Update duration
        audio_file.duration_seconds = result['duration']
        
        # Create segments
        segments_created = 0
        for segment in result.get('segments', []):
            TranscriptSegment.objects.create(
                audio_file=audio_file,
                start_time=segment['start'],
                end_time=segment['end'],
                text=segment['text'],
                language=result.get('language', 'ne'),
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
            'duration': result['duration'],
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