"""
ElevenLabs integration for audio translation and TTS
"""

import logging
import os
from io import BytesIO
import requests
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)


class ElevenLabsService:
    """Service for ElevenLabs API calls"""
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    def __init__(self):
        self.api_key = os.environ.get('ELEVENLABS_API_KEY')
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not set")
    
    def _get_headers(self):
        """Get request headers with API key"""
        return {
            'xi-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def text_to_speech(self, text, voice_id='21m00Tcm4TlvDq8ikWAM', model_id='eleven_monolingual_v1'):
        """
        Convert text to speech using ElevenLabs
        
        Args:
            text: Text to convert
            voice_id: ElevenLabs voice ID (default: Rachel)
            model_id: Model to use
            
        Returns:
            bytes: Audio data, or None on error
        """
        try:
            url = f"{self.BASE_URL}/text-to-speech/{voice_id}"
            
            payload = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            headers = self._get_headers()
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.content
            
        except requests.RequestException as e:
            logger.error(f"ElevenLabs TTS error: {str(e)}")
            return None
    
    def get_voices(self):
        """
        Get available voices from ElevenLabs
        
        Returns:
            list: List of available voices
        """
        try:
            url = f"{self.BASE_URL}/voices"
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('voices', [])
            
        except requests.RequestException as e:
            logger.error(f"ElevenLabs get voices error: {str(e)}")
            return []
    
    def get_voice_settings(self, voice_id):
        """
        Get settings for a specific voice
        
        Returns:
            dict: Voice settings
        """
        try:
            url = f"{self.BASE_URL}/voices/{voice_id}"
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"ElevenLabs get voice settings error: {str(e)}")
            return {}
    
    def estimate_price(self, text):
        """
        Estimate the cost of converting text to speech
        
        Args:
            text: Text to estimate
            
        Returns:
            dict: Cost information
        """
        try:
            url = f"{self.BASE_URL}/text-to-speech/estimate"
            
            payload = {"text": text}
            headers = self._get_headers()
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"ElevenLabs estimate price error: {str(e)}")
            return {"error": str(e)}


def generate_audio_from_text(text, voice_id='21m00Tcm4TlvDq8ikWAM', filename=None):
    """
    Convenience function to generate audio file from text
    
    Args:
        text: Text to convert
        voice_id: ElevenLabs voice ID
        filename: Filename for the ContentFile
        
    Returns:
        ContentFile: Django ContentFile with audio data, or None on error
    """
    service = ElevenLabsService()
    audio_data = service.text_to_speech(text, voice_id=voice_id)
    
    if audio_data:
        return ContentFile(audio_data, name=filename or 'translation.mp3')
    
    return None
