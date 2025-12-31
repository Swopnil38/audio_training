"""
URL configuration for training API.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    TrainingConfigViewSet,
    TrainingRunViewSet,
    ModelVersionViewSet,
    EvaluationResultViewSet,
)

# Create router
router = DefaultRouter()
router.register(r'configs', TrainingConfigViewSet, basename='trainingconfig')
router.register(r'runs', TrainingRunViewSet, basename='trainingrun')
router.register(r'models', ModelVersionViewSet, basename='modelversion')
router.register(r'evaluations', EvaluationResultViewSet, basename='evaluationresult')

urlpatterns = [
    path('', include(router.urls)),
]
