"""
Celery tasks for ASR processing - Multi-language support
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
_nepali_model = None
_nepali_processor = None
_whisper_model = None


@worker_process_init.connect
def preload_models(**kwargs):
    """Load models when worker starts"""
    global _nepali_model, _nepali_processor, _whisper_model
    
    # Load fine-tuned Nepali model
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    model_id = getattr(settings, 'WHISPER_MODEL_ID', '/home/docker/audio_training/models/whisper-small-nepali')
    device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
    
    logger.info(f"Preloading Nepali Whisper model: {model_id}")
    _nepali_processor = WhisperProcessor.from_pretrained(model_id)
    _nepali_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    _nepali_model.to(device)
    
    # Load original Whisper for English/Mixed
    import whisper
    logger.info("Preloading OpenAI Whisper model: small")
    _whisper_model = whisper.load_model("small", device=device)
    
    logger.info("All models loaded successfully!")


def transcribe_nepali(audio_path: str) -> dict:
    """Transcribe Nepali audio using fine-tuned model + transliteration"""
    global _nepali_model, _nepali_processor
    
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    
    device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
    
    # Load if not cached
    if _nepali_model is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        model_id = getattr(settings, 'WHISPER_MODEL_ID', '/home/docker/audio_training/models/whisper-small-nepali')
        _nepali_processor = WhisperProcessor.from_pretrained(model_id)
        _nepali_model = WhisperForConditionalGeneration.from_pretrained(model_id)
        _nepali_model.to(device)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    chunk_length = 30 * sr
    segments = []
    
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        start_time = i / sr
        end_time = min((i + chunk_length) / sr, duration)
        
        input_features = _nepali_processor(
            chunk, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = _nepali_model.generate(
                input_features,
                max_length=448,
                num_beams=5
            )
        
        text = _nepali_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        if text.strip():
            # Convert Roman to Devanagari
            devanagari_text = transliterate(text.strip(), sanscript.ITRANS, sanscript.DEVANAGARI)
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': devanagari_text
            })
    
    return {
        'segments': segments,
        'language': 'ne',
        'duration': duration
    }


def transcribe_english(audio_path: str) -> dict:
    """Transcribe English audio using OpenAI Whisper"""
    global _whisper_model
    
    import whisper
    
    if _whisper_model is None:
        device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
        _whisper_model = whisper.load_model("small", device=device)
    
    result = _whisper_model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        verbose=False
    )
    
    segments = []
    for segment in result.get('segments', []):
        if segment['text'].strip():
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
    
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    return {
        'segments': segments,
        'language': 'en',
        'duration': duration
    }


def transcribe_mixed(audio_path: str) -> dict:
    """Transcribe mixed Nepali+English using OpenAI Whisper with auto-detect"""
    global _whisper_model
    
    import whisper
    
    if _whisper_model is None:
        device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
        _whisper_model = whisper.load_model("small", device=device)
    
    # Auto-detect language, handles code-switching
    result = _whisper_model.transcribe(
        audio_path,
        language=None,  # Auto-detect
        task="transcribe",
        verbose=False
    )
    
    segments = []
    for segment in result.get('segments', []):
        if segment['text'].strip():
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
    
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    return {
        'segments': segments,
        'language': result.get('language', 'mixed'),
        'duration': duration
    }


@shared_task(bind=True, max_retries=3)
def transcribe_audio_task(self, audio_file_id: str, job_id: str):
    """
    Transcribe audio file based on selected language
    """
    from .models import AudioFile, TranscriptSegment, TranscriptionJob
    
    try:
        audio_file = AudioFile.objects.get(id=audio_file_id)
        job = TranscriptionJob.objects.get(id=job_id)
        
        job.status = TranscriptionJob.Status.PROCESSING
        job.started_at = timezone.now()
        job.save()
        
        logger.info(f"Starting transcription for {audio_file.original_filename} (language: {audio_file.language})")
        
        audio_path = audio_file.file.path
        logger.info(f"Transcribing: {audio_path}")
        
        # Choose transcription method based on language
        if audio_file.language == 'ne':
            result = transcribe_nepali(audio_path)
        elif audio_file.language == 'en':
            result = transcribe_english(audio_path)
        else:  # mixed
            result = transcribe_mixed(audio_path)
        
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
                language=result.get('language', audio_file.language),
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