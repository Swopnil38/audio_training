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
    path('list/', views.AudioListView.as_view(), name='audio-list'),
    path('editor/<uuid:audio_id>/', views.EditorView.as_view(), name='editor'),
    
    # API endpoints
    path('api/', include(router.urls)),
    path('api/corrections/bulk/', views.BulkCorrectionView.as_view(), name='bulk-corrections'),
    path('api/dashboard/', views.dashboard_stats, name='dashboard-stats'),
    path('api/export/', views.export_training_data, name='export-data'),
]
