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
    import os
    from dotenv import load_dotenv
    from io import BytesIO
    from elevenlabs.client import ElevenLabs
    import librosa

    load_dotenv()
    elevenlabs = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )

    # Read audio file as bytes
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Transcribe using ElevenLabs SDK

    transcription = elevenlabs.speech_to_text.convert(
        file=BytesIO(audio_data),
        model_id="scribe_v2",
        tag_audio_events=True,
        language_code=None,  # or set to "nep" for Nepali, "eng" for English, etc.
        diarize=True,
    )

    # The SDK returns a SpeechToTextChunkResponseModel object, not a dict
    text = getattr(transcription, "text", "")
    language = getattr(transcription, "language_code", "unknown")

    # Use librosa to get duration
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr

    return {
        'segments': [{
            'start': 0,
            'end': duration,
            'text': text
        }],
        'language': language,
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


# ===================
# Audio Chat Tasks
# ===================

@shared_task(bind=True, max_retries=3)
def process_audio_message(self, message_id: str, chat_id: str, group_name: str = None):
    """
    Process audio message: transcribe and translate
    
    Args:
        message_id: UUID of AudioChatMessage
        chat_id: UUID of AudioChat
        group_name: WebSocket group name for broadcasting updates
    """
    from .models import AudioChatMessage, AudioChat
    from .services.elevenlabs_service import ElevenLabsService, generate_audio_from_text
    from asgiref.sync import async_to_sync
    from channels.layers import get_channel_layer
    import json
    
    try:
        message = AudioChatMessage.objects.get(id=message_id)
        chat = AudioChat.objects.get(id=chat_id)
        
        # Update status
        message.status = AudioChatMessage.Status.TRANSCRIBING
        message.save()
        
        # Broadcast status update
        if group_name:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'chat_update',
                    'message_id': str(message.id),
                    'status': 'transcribing'
                }
            )
        
        # Transcribe audio using ElevenLabs
        service = ElevenLabsService()
        audio_path = message.audio_file.path
        
        try:
            from io import BytesIO
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Transcribe using ElevenLabs STT
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs()
            
            transcription = client.speech_to_text.convert(
                file=BytesIO(audio_data),
                model_id="scribe_v2",
                language_code=None,
            )
            
            original_text = getattr(transcription, 'text', '')
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            original_text = "[Transcription failed]"
        
        message.original_text = original_text
        message.status = AudioChatMessage.Status.TRANSLATING
        message.save()
        
        # Broadcast transcription
        if group_name:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'chat_update',
                    'message_id': str(message.id),
                    'status': 'translating',
                    'original_text': original_text
                }
            )
        
        # Translate text (if needed)
        translated_text = original_text  # Simple pass-through for now
        
        # You can add translation service here if needed
        # from google.cloud import translate_v2
        
        message.translated_text = translated_text
        message.save()
        
        # Generate audio from translated text using ElevenLabs TTS
        try:
            audio_file = generate_audio_from_text(translated_text)
            if audio_file:
                message.translated_audio = audio_file
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
        
        message.status = AudioChatMessage.Status.COMPLETED
        message.save()
        
        # Broadcast completion
        if group_name:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'chat_update',
                    'message_id': str(message.id),
                    'status': 'completed',
                    'translated_text': translated_text
                }
            )
        
        logger.info(f"Audio message {message_id} processed successfully")
        
    except AudioChatMessage.DoesNotExist:
        logger.error(f"AudioChatMessage {message_id} not found")
    except AudioChat.DoesNotExist:
        logger.error(f"AudioChat {chat_id} not found")
    except Exception as e:
        logger.error(f"Error processing audio message: {str(e)}")
        
        try:
            message = AudioChatMessage.objects.get(id=message_id)
            message.status = AudioChatMessage.Status.FAILED
            message.error_message = str(e)
            message.save()
        except Exception:
            pass
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))