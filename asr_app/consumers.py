"""
WebSocket consumers for audio chat
"""

import json
import asyncio
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

logger = logging.getLogger(__name__)


class AudioChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time audio chat"""
    
    async def connect(self):
        """Handle WebSocket connection"""
        self.chat_id = self.scope['url_route']['kwargs']['chat_id']
        self.user = self.scope['user']
        self.chat_group_name = f'audio_chat_{self.chat_id}'
        
        # Track continuous audio chunks by temp_id
        self.audio_buffer = {}  # {temp_id: [chunk1, chunk2, ...]}
        
        # Join the chat group
        await self.channel_layer.group_add(
            self.chat_group_name,
            self.channel_name
        )
        
        # Verify user has access to this chat
        if await self.user_has_access():
            await self.accept()
            logger.info(f"User {self.user.username} connected to chat {self.chat_id}")
        else:
            logger.warning(f"User {self.user.username} denied access to chat {self.chat_id}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        await self.channel_layer.group_discard(
            self.chat_group_name,
            self.channel_name
        )
        logger.info(f"User {self.user.username} disconnected from chat {self.chat_id}")
    
    async def receive(self, text_data):
        """Receive message from WebSocket"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'audio_message':
                await self.handle_audio_message(data)
            elif message_type == 'settings_update':
                await self.handle_settings_update(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            await self.send_error("Invalid JSON")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await self.send_error(f"Error: {str(e)}")
    
    async def handle_audio_message(self, data):
        """Handle incoming audio message"""
        try:
            # Extract audio data
            audio_base64 = data.get('audio')
            language = data.get('language', 'mixed')
            temp_id = data.get('temp_id')
            is_continuous = data.get('is_continuous', False)
            
            if not audio_base64:
                await self.send_error("No audio data provided")
                return
            
            # Accumulate chunks if continuous send
            if is_continuous and temp_id:
                if temp_id not in self.audio_buffer:
                    self.audio_buffer[temp_id] = []
                    logger.info(f"Starting new continuous message group: {temp_id}")
                
                self.audio_buffer[temp_id].append(audio_base64)
                logger.info(f"Buffered chunk for {temp_id}, total chunks: {len(self.audio_buffer[temp_id])}")
                
                # Send progress update to client
                await self.channel_layer.group_send(
                    self.chat_group_name,
                    {
                        'type': 'chat_update',
                        'message_id': temp_id,
                        'status': 'transcribing',
                        'original_text': f'(Listening... {len(self.audio_buffer[temp_id])} chunks)',
                    }
                )
                return
            
            # Final send - combine all buffered chunks + this final chunk
            if temp_id and temp_id in self.audio_buffer:
                # Combine all chunks
                self.audio_buffer[temp_id].append(audio_base64)
                combined_audio = self.combine_audio_chunks(self.audio_buffer[temp_id])
                logger.info(f"Final send for {temp_id}, combined {len(self.audio_buffer[temp_id])} chunks")
                del self.audio_buffer[temp_id]
            else:
                # No buffering, use as-is (single send without continuous mode)
                combined_audio = audio_base64
            
            # Create message record with combined audio
            message = await self.create_message(
                audio_base64=combined_audio,
                language=language,
                temp_id=temp_id
            )
            
            # Send notification to group
            await self.channel_layer.group_send(
                self.chat_group_name,
                {
                    'type': 'chat_message',
                    'message_id': str(message.id),
                    'temp_id': temp_id,
                    'role': 'user',
                    'status': 'transcribing',
                    'timestamp': message.created_at.isoformat()
                }
            )
            
            # Process audio asynchronously
            await self.process_message_async(message)
            
        except Exception as e:
            logger.error(f"Error handling audio message: {str(e)}")
            await self.send_error(f"Failed to process audio: {str(e)}")
    
    async def handle_settings_update(self, data):
        """Handle chat settings update"""
        try:
            await self.update_chat_settings(data)
            
            # Notify group
            await self.channel_layer.group_send(
                self.chat_group_name,
                {
                    'type': 'settings_changed',
                    'settings': data
                }
            )
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            await self.send_error(f"Failed to update settings: {str(e)}")
    
    async def chat_message(self, event):
        """Handle chat message event from group"""
        await self.send(text_data=json.dumps({
            'type': 'message',
            'data': event
        }))
    
    async def chat_update(self, event):
        """Handle chat update event"""
        await self.send(text_data=json.dumps({
            'type': 'update',
            'data': event
        }))
    
    async def settings_changed(self, event):
        """Handle settings change event"""
        await self.send(text_data=json.dumps({
            'type': 'settings',
            'data': event
        }))
    
    async def send_error(self, error_message):
        """Send error message to client"""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': error_message
        }))
    
    # Database operations
    
    def combine_audio_chunks(self, audio_chunks):
        """Combine multiple base64 audio chunks into one proper WAV file"""
        import base64
        import io
        import wave
        
        if not audio_chunks:
            return ""
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        try:
            # Decode all chunks
            decoded = [base64.b64decode(chunk) for chunk in audio_chunks]
            
            # Read all WAV files and combine audio data
            combined_frames = []
            sample_rate = None
            channels = None
            sample_width = None
            
            for chunk_data in decoded:
                try:
                    with wave.open(io.BytesIO(chunk_data), 'rb') as wav_file:
                        # Get WAV parameters from first chunk
                        if sample_rate is None:
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                            sample_rate = wav_file.getframerate()
                        
                        # Read and accumulate frames
                        frames = wav_file.readframes(wav_file.getnframes())
                        combined_frames.append(frames)
                except Exception as e:
                    logger.warning(f"Error reading WAV chunk: {e}, skipping")
                    continue
            
            if not combined_frames:
                logger.warning("No valid WAV chunks to combine")
                return audio_chunks[0]
            
            # Create new WAV file with combined audio
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b''.join(combined_frames))
            
            # Return combined audio as base64
            combined_data = output.getvalue()
            return base64.b64encode(combined_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error combining audio chunks: {e}, falling back to simple concat")
            # Fallback: simple concatenation
            combined = decoded[0]
            for chunk in decoded[1:]:
                if len(chunk) > 44:
                    combined += chunk[44:]
                else:
                    combined += chunk
            return base64.b64encode(combined).decode('utf-8')
    
    @database_sync_to_async
    def user_has_access(self):
        """Check if user has access to this chat"""
        from .models import AudioChat
        try:
            AudioChat.objects.get(id=self.chat_id, user=self.user)
            return True
        except AudioChat.DoesNotExist:
            return False
    
    @database_sync_to_async
    def create_message(self, audio_base64, language, temp_id=None):
        """Create audio chat message"""
        import base64
        from django.core.files.base import ContentFile
        from .models import AudioChat, AudioChatMessage
        
        # Decode and save audio
        audio_data = base64.b64decode(audio_base64)
        audio_file = ContentFile(audio_data, name='message.wav')
        
        chat = AudioChat.objects.get(id=self.chat_id)
        
        message = AudioChatMessage.objects.create(
            chat=chat,
            role='user',
            status='transcribing',
            audio_file=audio_file,
            temp_id=temp_id  # Store temp_id to link with frontend
        )
        
        return message
    
    @database_sync_to_async
    def update_chat_settings(self, settings):
        """Update chat settings"""
        from .models import AudioChat
        chat = AudioChat.objects.get(id=self.chat_id)
        
        if 'target_language' in settings:
            chat.target_language = settings['target_language']
        if 'auto_play_translation' in settings:
            chat.auto_play_translation = settings['auto_play_translation']
        
        chat.save()
    
    async def process_message_async(self, message):
        """Process message asynchronously"""
        from .tasks import process_audio_message
        # Use Celery task to process the audio
        process_audio_message.delay(str(message.id), str(self.chat_id), self.chat_group_name)
