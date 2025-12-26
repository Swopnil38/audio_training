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
    
    model_id = getattr(settings, 'WHISPER_MODEL_ID', 'Dragneel/whisper-small-nepali')
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
        
        model_id = getattr(settings, 'WHISPER_MODEL_ID', 'Dragneel/whisper-small-nepali')
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
        
        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=5,
                language="ne",  # Nepali
                task="transcribe"
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
        
        # Generate - removed language argument
        with torch.no_grad():
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="ne", task="transcribe")
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=5,
                forced_decoder_ids=forced_decoder_ids
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


@shared_task
def generate_embeddings_task(segment_id: str):
    """Generate embeddings for a transcript segment"""
    from .models import TranscriptSegment
    from sentence_transformers import SentenceTransformer
    
    try:
        segment = TranscriptSegment.objects.get(id=segment_id)
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        text = segment.final_text
        embedding = model.encode(text)
        
        segment.embedding = embedding.tolist()
        segment.save()
        
        logger.info(f"Generated embedding for segment {segment_id}")
        return {'status': 'success', 'segment_id': str(segment_id)}
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}