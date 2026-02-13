"""
WebSocket consumers for audio chat
Optimized for ChatGPT-style pause detection (5-second auto-send)
"""

import json
import asyncio
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

logger = logging.getLogger(__name__)


class AudioChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time audio chat with pause detection"""
    
    async def connect(self):
        """Handle WebSocket connection"""
        # Support both unified chat and chat_id-specific chats
        url_kwargs = self.scope['url_route'].get('kwargs', {})
        self.user = self.scope['user']
        
        # If chat_id provided, use it; otherwise use unified chat
        if 'chat_id' in url_kwargs:
            self.chat_id = url_kwargs.get('chat_id')
        else:
            # Get or create unified chat for this user
            unified_chat = await self.get_or_create_unified_chat()
            self.chat_id = str(unified_chat.id)
        
        self.chat_group_name = f'audio_chat_{self.chat_id}'
        
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
    
    async def receive(self, text_data=None, bytes_data=None):
        """Receive message from WebSocket - handles both text and binary data"""
        try:
            # Handle text data (JSON messages)
            if text_data:
                data = json.loads(text_data)
                message_type = data.get('type')
                
                if message_type == 'audio_message':
                    await self.handle_audio_message(data)
                elif message_type == 'settings_update':
                    await self.handle_settings_update(data)
                else:
                    logger.warning(f"Unknown message type: {message_type}")
            
            # Handle binary data (raw audio)
            elif bytes_data:
                logger.info(f"Received {len(bytes_data)} bytes of audio data")
                # Binary audio data - convert to base64 and process
                import base64
                audio_base64 = base64.b64encode(bytes_data).decode('utf-8')
                await self.handle_audio_message({
                    'type': 'audio_message',
                    'audio': audio_base64,
                    'language': 'mixed'
                })
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            await self.send_error("Invalid JSON")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await self.send_error(f"Error: {str(e)}")
    
    async def handle_audio_message(self, data):
        """
        Handle incoming audio message.
        Frontend sends complete audio after 5-second pause detection.
        """
        try:
            # Extract audio data
            audio_base64 = data.get('audio')
            language = data.get('language', 'mixed')
            temp_id = data.get('temp_id')
            
            if not audio_base64:
                await self.send_error("No audio data provided")
                return
            
            logger.info(f"Received audio message - temp_id: {temp_id}, language: {language}")
            
            # Create message record
            message = await self.create_message(
                audio_base64=audio_base64,
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
            
            # Process audio asynchronously (transcription + translation)
            await self.process_message_async(message)
            
        except Exception as e:
            logger.error(f"Error handling audio message: {str(e)}", exc_info=True)
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
    
    @database_sync_to_async
    def get_or_create_unified_chat(self):
        """Get or create a unified chat for the current user"""
        from .models import AudioChat
        # Resolve the lazy user object
        user = self.user
        if hasattr(user, '_wrapped'):
            # It's a lazy object, force evaluation
            user = user._wrapped
        
        chat, created = AudioChat.objects.get_or_create(
            user=user,
            title='Unified Chat',
            defaults={
                'source_language': 'mixed',
                'target_language': 'en',
                'auto_play_translation': True
            }
        )
        if created:
            logger.info(f"Created unified chat {chat.id} for user {user.username}")
        return chat
    
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
        
        logger.info(f"Created message {message.id} with temp_id: {temp_id}")
        return message
    
    @database_sync_to_async
    def update_chat_settings(self, settings):
        """Update chat settings"""
        from .models import AudioChat
        chat = AudioChat.objects.get(id=self.chat_id)
        
        if 'source_language' in settings:
            chat.source_language = settings['source_language']
        if 'target_language' in settings:
            chat.target_language = settings['target_language']
        if 'auto_play_translation' in settings:
            chat.auto_play_translation = settings['auto_play_translation']
        
        chat.save()
        logger.info(f"Updated chat settings: {settings}")
    
    async def process_message_async(self, message):
        """Process message asynchronously using Celery"""
        from .tasks import process_audio_message
        # Use Celery task to process the audio (transcription + translation)
        process_audio_message.delay(
            str(message.id), 
            str(self.chat_id), 
            self.chat_group_name
        )
        logger.info(f"Queued processing task for message {message.id}")