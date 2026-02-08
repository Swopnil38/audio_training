"""
API Views for ASR System
"""

from rest_framework import viewsets, status, generics
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.db.models import Count, Q
from django.views.generic import TemplateView, DetailView

from .models import AudioFile, TranscriptSegment, Correction, TranscriptionJob


# ===================
# Template Views
# ===================

class UploadView(TemplateView):
    """Upload page view"""
    template_name = 'asr_app/upload.html'


class AudioListView(TemplateView):
    """Audio list page view"""
    template_name = 'asr_app/audio_list.html'


class EditorView(DetailView):
    """Transcription editor page view"""
    model = AudioFile
    template_name = 'asr_app/editor.html'
    context_object_name = 'audio_file'
    pk_url_kwarg = 'audio_id'


# ===================
# API Views
# ===================
from .serializers import (
    AudioFileListSerializer,
    AudioFileDetailSerializer,
    AudioFileUploadSerializer,
    TranscriptSegmentSerializer,
    SegmentCorrectionSerializer,
    TranscriptionJobSerializer,
    CorrectionSerializer,
    BulkCorrectionSerializer,
    AudioChatSerializer,
    AudioChatCreateSerializer,
    AudioChatMessageSerializer,
)
from .tasks import transcribe_audio_task


class AudioFileViewSet(viewsets.ModelViewSet):
    """
    ViewSet for audio file operations
    """
    queryset = AudioFile.objects.all()
    parser_classes = [MultiPartParser, FormParser]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return AudioFileListSerializer
        elif self.action == 'create':
            return AudioFileUploadSerializer
        return AudioFileDetailSerializer
    
    def create(self, request, *args, **kwargs):
        """Upload audio and queue for transcription"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Save audio file
        audio_file = serializer.save(
            uploaded_by=request.user if request.user.is_authenticated else None
        )
        
        # Queue transcription task
        job = TranscriptionJob.objects.create(audio_file=audio_file)
        task = transcribe_audio_task.delay(str(audio_file.id), str(job.id))
        job.celery_task_id = task.id
        job.save()
        
        # Update audio status
        audio_file.status = AudioFile.Status.PROCESSING
        audio_file.save()
        
        return Response({
            'audio_file': AudioFileDetailSerializer(audio_file, context={'request': request}).data,
            'job': TranscriptionJobSerializer(job).data
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """Get processing status"""
        audio_file = self.get_object()
        latest_job = audio_file.jobs.order_by('-created_at').first()
        
        return Response({
            'audio_status': audio_file.status,
            'job': TranscriptionJobSerializer(latest_job).data if latest_job else None
        })
    
    @action(detail=True, methods=['post'])
    def retranscribe(self, request, pk=None):
        """Re-run transcription"""
        audio_file = self.get_object()
        
        # Clear existing segments
        audio_file.segments.all().delete()
        
        # Queue new job
        job = TranscriptionJob.objects.create(audio_file=audio_file)
        task = transcribe_audio_task.delay(str(audio_file.id), str(job.id))
        job.celery_task_id = task.id
        job.save()
        
        audio_file.status = AudioFile.Status.PROCESSING
        audio_file.save()
        
        return Response({
            'message': 'Retranscription queued',
            'job': TranscriptionJobSerializer(job).data
        })
    
    @action(detail=True, methods=['post'])
    def approve(self, request, pk=None):
        """Mark audio as approved for training"""
        audio_file = self.get_object()
        
        if audio_file.status not in [AudioFile.Status.TRANSCRIBED, AudioFile.Status.CORRECTED]:
            return Response(
                {'error': 'Audio must be transcribed or corrected before approval'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        audio_file.status = AudioFile.Status.APPROVED
        audio_file.save()
        
        return Response({'message': 'Audio approved', 'status': audio_file.status})


class TranscriptSegmentViewSet(viewsets.ModelViewSet):
    """
    ViewSet for transcript segment operations
    """
    serializer_class = TranscriptSegmentSerializer
    
    def get_queryset(self):
        queryset = TranscriptSegment.objects.all()
        
        # Filter by audio file
        audio_id = self.request.query_params.get('audio_id')
        if audio_id:
            queryset = queryset.filter(audio_file_id=audio_id)
        
        # Filter by correction status
        corrected = self.request.query_params.get('corrected')
        if corrected is not None:
            queryset = queryset.filter(is_corrected=corrected.lower() == 'true')
        
        return queryset.order_by('audio_file', 'start_time')
    
    @action(detail=True, methods=['post'])
    def correct(self, request, pk=None):
        """Submit correction for a segment"""
        segment = self.get_object()
        serializer = SegmentCorrectionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        
        # Create correction record
        Correction.objects.create(
            segment=segment,
            original_text=segment.text if not segment.corrected_text else segment.corrected_text,
            corrected_text=data['corrected_text'],
            correction_type=data.get('correction_type', Correction.CorrectionType.WORD),
            notes=data.get('notes', ''),
            corrected_by=request.user if request.user.is_authenticated else None
        )
        
        # Update segment
        segment.corrected_text = data['corrected_text']
        segment.is_corrected = True
        segment.corrected_at = timezone.now()
        segment.corrected_by = request.user if request.user.is_authenticated else None
        segment.save()
        
        # Update audio file status
        audio_file = segment.audio_file
        if audio_file.status == AudioFile.Status.TRANSCRIBED:
            audio_file.status = AudioFile.Status.CORRECTED
            audio_file.save()
        
        return Response(TranscriptSegmentSerializer(segment).data)


class BulkCorrectionView(generics.CreateAPIView):
    """
    Submit multiple corrections at once
    """
    serializer_class = BulkCorrectionSerializer
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        corrections = serializer.validated_data['corrections']
        results = []
        
        for item in corrections:
            try:
                segment = TranscriptSegment.objects.get(id=item['segment_id'])
                
                # Create correction record
                Correction.objects.create(
                    segment=segment,
                    original_text=segment.text if not segment.corrected_text else segment.corrected_text,
                    corrected_text=item['corrected_text'],
                    correction_type=item.get('correction_type', Correction.CorrectionType.WORD),
                    notes=item.get('notes', ''),
                    corrected_by=request.user if request.user.is_authenticated else None
                )
                
                # Update segment
                segment.corrected_text = item['corrected_text']
                segment.is_corrected = True
                segment.corrected_at = timezone.now()
                segment.save()
                
                results.append({
                    'segment_id': str(segment.id),
                    'status': 'success'
                })
                
            except TranscriptSegment.DoesNotExist:
                results.append({
                    'segment_id': item['segment_id'],
                    'status': 'error',
                    'message': 'Segment not found'
                })
            except Exception as e:
                results.append({
                    'segment_id': item.get('segment_id'),
                    'status': 'error',
                    'message': str(e)
                })
        
        return Response({'results': results})


class TranscriptionJobViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing transcription jobs
    """
    queryset = TranscriptionJob.objects.all()
    serializer_class = TranscriptionJobSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        audio_id = self.request.query_params.get('audio_id')
        if audio_id:
            queryset = queryset.filter(audio_file_id=audio_id)
        
        return queryset


@api_view(['GET'])
def dashboard_stats(request):
    """
    Dashboard statistics
    """
    total_audio = AudioFile.objects.count()
    total_segments = TranscriptSegment.objects.count()
    corrected_segments = TranscriptSegment.objects.filter(is_corrected=True).count()
    
    status_counts = dict(
        AudioFile.objects.values('status')
        .annotate(count=Count('id'))
        .values_list('status', 'count')
    )
    
    language_counts = dict(
        AudioFile.objects.values('language')
        .annotate(count=Count('id'))
        .values_list('language', 'count')
    )
    
    # Recent activity
    recent_uploads = AudioFile.objects.order_by('-created_at')[:5]
    recent_corrections = Correction.objects.order_by('-created_at')[:10]
    
    return Response({
        'summary': {
            'total_audio_files': total_audio,
            'total_segments': total_segments,
            'corrected_segments': corrected_segments,
            'correction_rate': round(corrected_segments / total_segments * 100, 1) if total_segments > 0 else 0,
        },
        'by_status': status_counts,
        'by_language': language_counts,
        'recent_uploads': AudioFileListSerializer(recent_uploads, many=True).data,
        'recent_corrections': CorrectionSerializer(recent_corrections, many=True).data,
    })


@api_view(['GET'])
def export_training_data(request):
    """
    Export corrected data for training
    """
    format_type = request.query_params.get('format', 'json')
    include_uncorrected = request.query_params.get('include_uncorrected', 'false').lower() == 'true'
    language = request.query_params.get('language', 'all')
    
    # Get segments
    segments = TranscriptSegment.objects.select_related('audio_file')
    
    if not include_uncorrected:
        segments = segments.filter(is_corrected=True)
    
    if language != 'all':
        segments = segments.filter(audio_file__language=language)
    
    # Build export data
    export_data = []
    for segment in segments:
        export_data.append({
            'audio_file': str(segment.audio_file.id),
            'audio_path': segment.audio_file.file.name if segment.audio_file.file else None,
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'original_text': segment.text,
            'corrected_text': segment.corrected_text or segment.text,
            'is_corrected': segment.is_corrected,
            'language': segment.audio_file.language,
        })
    
    return Response({
        'count': len(export_data),
        'format': format_type,
        'data': export_data
    })


# ===================
# Audio Chat Views
# ===================

class AudioChatView(TemplateView):
    """Audio chat page view"""
    template_name = 'asr_app/audio_chat.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['chat_id'] = kwargs.get('chat_id')
        return context


class AudioChatsListView(TemplateView):
    """List audio chats page view"""
    template_name = 'asr_app/chats.html'


class AudioChatListView(generics.ListCreateAPIView):
    """List and create audio chats"""
    
    serializer_class = None  # Will be set in methods
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            from .serializers import AudioChatCreateSerializer
            return AudioChatCreateSerializer
        else:
            from .serializers import AudioChatSerializer
            return AudioChatSerializer
    
    def get_queryset(self):
        from .models import AudioChat
        if self.request.user.is_authenticated:
            return AudioChat.objects.filter(user=self.request.user)
        return AudioChat.objects.none()
    
    def perform_create(self, serializer):
        from .models import AudioChat
        data = serializer.validated_data
        AudioChat.objects.create(
            user=self.request.user,
            title=data.get('title', ''),
            source_language=data.get('source_language', 'mixed'),
            target_language=data.get('target_language', 'en'),
            auto_play_translation=data.get('auto_play_translation', True)
        )


class AudioChatDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, and delete audio chat"""
    
    from .serializers import AudioChatSerializer
    serializer_class = AudioChatSerializer
    
    def get_queryset(self):
        from .models import AudioChat
        if self.request.user.is_authenticated:
            return AudioChat.objects.filter(user=self.request.user)
        return AudioChat.objects.none()


class AudioChatMessagesView(generics.ListAPIView):
    """Get messages for audio chat"""
    
    from .serializers import AudioChatMessageSerializer
    serializer_class = AudioChatMessageSerializer
    
    def get_queryset(self):
        from .models import AudioChatMessage, AudioChat
        chat_id = self.kwargs.get('chat_id')
        try:
            chat = AudioChat.objects.get(id=chat_id, user=self.request.user)
            return chat.messages.all()
        except AudioChat.DoesNotExist:
            return AudioChatMessage.objects.none()
