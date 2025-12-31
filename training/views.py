"""
REST API views for training management.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny

from .models import TrainingConfig, TrainingRun, ModelVersion, EvaluationResult
from .serializers import (
    TrainingConfigSerializer,
    TrainingRunSerializer,
    TrainingRunListSerializer,
    ModelVersionSerializer,
    ModelVersionListSerializer,
    EvaluationResultSerializer,
)
from .tasks import run_training_pipeline, cancel_training, evaluate_model


class TrainingConfigViewSet(viewsets.ModelViewSet):
    """
    API endpoints for TrainingConfig.

    list: Get all training configurations
    retrieve: Get a specific configuration
    create: Create a new configuration
    update: Update a configuration
    destroy: Delete a configuration
    """

    queryset = TrainingConfig.objects.all()
    serializer_class = TrainingConfigSerializer
    permission_classes = [AllowAny]  # Change to IsAuthenticated in production

    def get_queryset(self):
        """Filter queryset"""
        queryset = super().get_queryset()

        # Filter by base model
        base_model = self.request.query_params.get('base_model')
        if base_model:
            queryset = queryset.filter(base_model=base_model)

        return queryset

    @action(detail=True, methods=['post'])
    def duplicate(self, request, pk=None):
        """
        Duplicate a training configuration.
        POST /api/training/configs/{id}/duplicate/
        """
        config = self.get_object()

        # Create a copy
        config.pk = None
        config.name = f"{config.name} (Copy)"
        config.save()

        serializer = self.get_serializer(config)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class TrainingRunViewSet(viewsets.ModelViewSet):
    """
    API endpoints for TrainingRun.

    list: Get all training runs
    retrieve: Get a specific run with full details
    create: Create a new run (use /start/ instead)
    """

    queryset = TrainingRun.objects.all()
    permission_classes = [AllowAny]  # Change to IsAuthenticated in production

    def get_serializer_class(self):
        """Use different serializers for list vs detail"""
        if self.action == 'list':
            return TrainingRunListSerializer
        return TrainingRunSerializer

    def get_queryset(self):
        """Filter queryset"""
        queryset = super().get_queryset()

        # Filter by status
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)

        # Filter by config
        config_id = self.request.query_params.get('config_id')
        if config_id:
            queryset = queryset.filter(config_id=config_id)

        return queryset

    @action(detail=False, methods=['post'])
    def start(self, request):
        """
        Start a new training run.

        POST /api/training/runs/start/
        Body: {"config_id": 1}

        Returns:
            TrainingRun object with status='pending'
        """
        config_id = request.data.get('config_id')

        if not config_id:
            return Response(
                {'error': 'config_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            config = TrainingConfig.objects.get(id=config_id)
        except TrainingConfig.DoesNotExist:
            return Response(
                {'error': f'TrainingConfig with id={config_id} not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Check if there's already a running training
        active_runs = TrainingRun.objects.filter(
            status__in=['pending', 'preparing', 'training', 'evaluating', 'exporting']
        ).count()

        if active_runs > 0:
            return Response(
                {
                    'error': 'A training run is already in progress. '
                             'Please wait for it to complete or cancel it first.'
                },
                status=status.HTTP_409_CONFLICT
            )

        # Create training run
        run = TrainingRun.objects.create(config=config)

        # Start async task
        task = run_training_pipeline.delay(run.id)
        run.celery_task_id = task.id
        run.save()

        serializer = TrainingRunSerializer(run)
        return Response(
            serializer.data,
            status=status.HTTP_201_CREATED
        )

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """
        Cancel a running training.

        POST /api/training/runs/{id}/cancel/
        """
        run = self.get_object()

        if run.status not in ['pending', 'preparing', 'training', 'evaluating']:
            return Response(
                {
                    'error': f'Cannot cancel run with status={run.status}. '
                             'Only pending/preparing/training/evaluating runs can be cancelled.'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Cancel task
        cancel_training.delay(run.id)

        return Response({
            'message': 'Cancellation requested',
            'run_id': run.id,
        })

    @action(detail=True, methods=['get'])
    def progress(self, request, pk=None):
        """
        Get training progress.

        GET /api/training/runs/{id}/progress/

        Returns:
            Progress information (epoch, step, metrics)
        """
        run = self.get_object()

        progress = {
            'run_id': run.id,
            'status': run.status,
            'current_epoch': run.current_epoch,
            'current_step': run.current_step,
            'total_steps': run.total_steps,
            'progress_pct': (
                (run.current_step / run.total_steps * 100)
                if run.total_steps > 0 else 0
            ),
        }

        return Response(progress)

    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        """
        Get training logs.

        GET /api/training/runs/{id}/logs/
        """
        run = self.get_object()

        return Response({
            'run_id': run.id,
            'logs': run.logs,
            'error_message': run.error_message,
        })


class ModelVersionViewSet(viewsets.ModelViewSet):
    """
    API endpoints for ModelVersion.

    list: Get all model versions
    retrieve: Get a specific version
    create: Create a new version
    update: Update a version
    destroy: Delete a version
    """

    queryset = ModelVersion.objects.all()
    permission_classes = [AllowAny]  # Change to IsAuthenticated in production

    def get_serializer_class(self):
        """Use different serializers for list vs detail"""
        if self.action == 'list':
            return ModelVersionListSerializer
        return ModelVersionSerializer

    def get_queryset(self):
        """Filter queryset"""
        queryset = super().get_queryset()

        # Filter by active status
        is_active = self.request.query_params.get('is_active')
        if is_active is not None:
            queryset = queryset.filter(is_active=(is_active.lower() == 'true'))

        return queryset

    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """
        Set this model as the active deployment.

        POST /api/training/models/{id}/activate/

        This will:
        1. Deactivate all other models
        2. Set this model as active
        """
        version = self.get_object()

        # Deactivate all others
        ModelVersion.objects.filter(is_active=True).update(is_active=False)

        # Activate this one
        version.is_active = True
        version.save()

        serializer = self.get_serializer(version)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def deactivate(self, request, pk=None):
        """
        Deactivate this model.

        POST /api/training/models/{id}/deactivate/
        """
        version = self.get_object()
        version.is_active = False
        version.save()

        serializer = self.get_serializer(version)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def active(self, request):
        """
        Get the currently active model.

        GET /api/training/models/active/
        """
        try:
            version = ModelVersion.objects.get(is_active=True)
            serializer = self.get_serializer(version)
            return Response(serializer.data)
        except ModelVersion.DoesNotExist:
            return Response(
                {'error': 'No active model found'},
                status=status.HTTP_404_NOT_FOUND
            )

    @action(detail=True, methods=['post'])
    def evaluate(self, request, pk=None):
        """
        Trigger evaluation for this model.

        POST /api/training/models/{id}/evaluate/
        """
        version = self.get_object()

        # Start evaluation task
        task = evaluate_model.delay(version.id)

        return Response({
            'message': 'Evaluation started',
            'version': version.version,
            'task_id': task.id,
        })


class EvaluationResultViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoints for EvaluationResult (read-only).

    list: Get all evaluation results
    retrieve: Get a specific evaluation result
    """

    queryset = EvaluationResult.objects.all()
    serializer_class = EvaluationResultSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        """Filter queryset"""
        queryset = super().get_queryset()

        # Filter by training run
        run_id = self.request.query_params.get('run_id')
        if run_id:
            queryset = queryset.filter(training_run_id=run_id)

        # Filter by test set
        test_set = self.request.query_params.get('test_set')
        if test_set:
            queryset = queryset.filter(test_set=test_set)

        return queryset
