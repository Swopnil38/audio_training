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
    # First detect language to ensure it's English or Nepali
    transcription = elevenlabs.speech_to_text.convert(
        file=BytesIO(audio_data),
        model_id="scribe_v2",
        tag_audio_events=True,
        language_code=None,  # Auto-detect first
        diarize=True,
    )

    # The SDK returns a SpeechToTextChunkResponseModel object, not a dict
    text = getattr(transcription, "text", "")
    detected_language = getattr(transcription, "language_code", "unknown")
    
    # Filter: Only allow English (eng) and Nepali (nep)
    allowed_languages = ['eng', 'nep', 'en', 'ne']
    
    if detected_language and detected_language.lower() not in allowed_languages:
        logger.warning(f"Language {detected_language} not allowed. Only English and Nepali supported. Retranscribing as English.")
        # Re-transcribe with English as default fallback
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        transcription = elevenlabs.speech_to_text.convert(
            file=BytesIO(audio_data),
            model_id="scribe_v2",
            tag_audio_events=True,
            language_code="eng",  # Fallback to English
            diarize=True,
        )
        text = getattr(transcription, "text", "")
        detected_language = "eng"
    
    language = detected_language

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

@shared_task(bind=True, max_retries=2, time_limit=120)
def process_audio_message(self, message_id: str, chat_id: str, group_name: str = None):
    """
    REAL-TIME STREAMING: Use ElevenLabs WebSocket streaming for instant audio
    
    Args:
        message_id: UUID of AudioChatMessage
        chat_id: UUID of AudioChat
        group_name: WebSocket group name for broadcasting updates
    """
    from .models import AudioChatMessage, AudioChat
    from .services.streaming_service import RealtimeStreamingService
    from asgiref.sync import async_to_sync
    from channels.layers import get_channel_layer
    from io import BytesIO
    from elevenlabs.client import ElevenLabs
    from django.core.files.base import ContentFile
    
    try:
        message = AudioChatMessage.objects.get(id=message_id)
        chat = AudioChat.objects.get(id=chat_id)
        
        # Mark as processing
        message.status = AudioChatMessage.Status.TRANSCRIBING
        message.save(update_fields=['status'])
        
        audio_path = message.audio_file.path
        
        # FAST: Transcribe immediately
        try:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            client = ElevenLabs()
            transcription = client.speech_to_text.convert(
                file=BytesIO(audio_data),
                model_id="scribe_v2",
                language_code=None,  # Auto-detect first
            )
            
            # Language filtering: Only allow English and Nepali
            detected_language = getattr(transcription, 'language_code', 'unknown')
            allowed_languages = ['eng', 'nep', 'en', 'ne']
            
            if detected_language and detected_language.lower() not in allowed_languages:
                logger.warning(f"Detected language {detected_language} not allowed. Retranscribing as English.")
                # Retry with English as default
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                transcription = client.speech_to_text.convert(
                    file=BytesIO(audio_data),
                    model_id="scribe_v2",
                    language_code="eng",  # Force English if language not allowed
                )
            
            original_text = getattr(transcription, 'text', '') or "[No speech detected]"
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            original_text = "[Transcription failed]"
        
        message.original_text = original_text
        message.translated_text = original_text
        message.status = AudioChatMessage.Status.COMPLETED
        message.save(update_fields=['original_text', 'translated_text', 'status'])
        
        # Broadcast transcription instantly
        if group_name:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'chat_update',
                    'message_id': str(message.id),
                    'status': 'completed',
                    'original_text': original_text,
                    'translated_text': original_text
                }
            )
        
        # STREAMING: Generate TTS in background with real-time streaming chunks
        def generate_streaming_tts():
            try:
                streaming_service = RealtimeStreamingService()
                
                # Use streaming to get chunks as they're generated
                audio_chunks = []
                chunk_count = 0
                max_wait_time = 30  # Max 30 seconds to get TTS
                
                import time
                start_time = time.time()
                
                for chunk in streaming_service.stream_text_to_speech(original_text):
                    if chunk:
                        audio_chunks.append(chunk)
                        chunk_count += 1
                    
                    # Timeout protection
                    if time.time() - start_time > max_wait_time:
                        logger.warning(f"TTS streaming timeout for message {message_id}")
                        break
                
                if audio_chunks:
                    # Combine all chunks into final audio file
                    complete_audio = b''.join(audio_chunks)
                    audio_file = ContentFile(complete_audio, name='translation.mp3')
                    message.translated_audio = audio_file
                    message.save(update_fields=['translated_audio'])
                    
                    logger.info(f"Streamed {chunk_count} audio chunks for message {message_id}")
                    
                    # Broadcast TTS ready
                    if group_name:
                        channel_layer = get_channel_layer()
                        async_to_sync(channel_layer.group_send)(
                            group_name,
                            {
                                'type': 'chat_update',
                                'message_id': str(message.id),
                                'status': 'audio_ready',
                                'translated_audio_url': message.translated_audio.url if message.translated_audio else None
                            }
                        )
                else:
                    logger.warning(f"No TTS chunks received for message {message_id}")
                    
            except Exception as e:
                logger.error(f"Streaming TTS error: {str(e)}")
                # Don't crash - TTS is optional, transcription was successful
        
        # Start TTS streaming in background with timeout
        import threading
        tts_thread = threading.Thread(target=generate_streaming_tts, daemon=True)
        tts_thread.daemon = True
        tts_thread.start()
        
        logger.info(f"Audio message {message_id} processed (streaming in background)")
        
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
            message.save(update_fields=['status', 'error_message'])
            
            # Notify client of failure
            if group_name:
                channel_layer = get_channel_layer()
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        'type': 'chat_update',
                        'message_id': str(message.id),
                        'status': 'failed',
                        'error': str(e)
                    }
                )
        except Exception:
            pass
        
        raise self.retry(exc=e, countdown=10 * (2 ** self.request.retries))