"""
ASR App configuration
"""

from django.apps import AppConfig


class AsrAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'asr_app'
    verbose_name = 'ASR System'
