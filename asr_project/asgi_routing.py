"""
ASGI routing for WebSocket connections
"""

from django.urls import re_path
from asr_app.consumers import AudioChatConsumer

websocket_urlpatterns = [
    re_path(r'ws/audio-chat/(?P<chat_id>[a-f0-9-]+)/$', AudioChatConsumer.as_asgi()),
]
