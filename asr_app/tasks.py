"""
Celery tasks for ASR processing - Using Fine-tuned Nepali Whisper
"""

import os
import logging
import torch
import librosa
from celery import shared_task
from celery.signals import worker_process_init
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# Global model cache
_whisper_model = None
_whisper_processor = None


@worker_process_init.connect
def preload_whisper_model(**kwargs):
    """Load Whisper model when worker starts"""
    global _whisper_model, _whisper_processor
    
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    model_id = getattr(settings, 'WHISPER_MODEL_ID', '/home/docker/audio_training/models/whisper-small-nepali')
    device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
    
    logger.info(f"Preloading Whisper model: {model_id} on {device}")
    
    _whisper_processor = WhisperProcessor.from_pretrained(model_id)
    _whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    _whisper_model.to(device)
    
    logger.info("Whisper model loaded successfully!")


def get_whisper_model():
    """Get cached model or load if not available"""
    global _whisper_model, _whisper_processor
    
    if _whisper_model is None or _whisper_processor is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        model_id = getattr(settings, 'WHISPER_MODEL_ID', '/home/docker/audio_training/models/whisper-small-nepali')
        device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
        
        _whisper_processor = WhisperProcessor.from_pretrained(model_id)
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
        _whisper_model.to(device)
    
    return _whisper_model, _whisper_processor


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using fine-tuned Nepali Whisper"""
    
    model, processor = get_whisper_model()
    device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    # Process in chunks for long audio (30 sec chunks)
    chunk_length = 30 * sr  # 30 seconds
    segments = []
    
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        start_time = i / sr
        end_time = min((i + chunk_length) / sr, duration)
        
        # Prepare input
        input_features = processor(
            chunk, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate without language argument (use default from model)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=5
            )
        
        # Decode
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        if text.strip():
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': text.strip()
            })
    
    return {
        'segments': segments,
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