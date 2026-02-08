"""
URL routing for ASR API
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'audio', views.AudioFileViewSet, basename='audio')
router.register(r'segments', views.TranscriptSegmentViewSet, basename='segments')
router.register(r'jobs', views.TranscriptionJobViewSet, basename='jobs')

urlpatterns = [
    # Template views
    path('', views.UploadView.as_view(), name='upload'),
    path('list/', views.AudioListView.as_view(), name='audiofile-list'),
    path('editor/<uuid:audio_id>/', views.EditorView.as_view(), name='editor'),
    path('chat/<uuid:chat_id>/', views.AudioChatView.as_view(), name='audio-chat'),
    path('chats/', views.AudioChatsListView.as_view(), name='chats-list'),
    
    # API endpoints
    path('api/', include(router.urls)),
    path('api/corrections/bulk/', views.BulkCorrectionView.as_view(), name='bulk-corrections'),
    path('api/dashboard/', views.dashboard_stats, name='dashboard-stats'),
    path('api/export/', views.export_training_data, name='export-data'),
    
    # Audio Chat API
    path('api/chats/', views.AudioChatListView.as_view(), name='chat-list'),
    path('api/chats/<uuid:pk>/', views.AudioChatDetailView.as_view(), name='chat-detail'),
    path('api/chats/<uuid:chat_id>/messages/', views.AudioChatMessagesView.as_view(), name='chat-messages'),
]
