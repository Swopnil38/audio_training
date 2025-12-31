"""
Django management command to export a trained model.
"""
import os
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from training.models import ModelVersion
from training.services.exporter import ModelExporter


class Command(BaseCommand):
    help = 'Export a model to optimized format'

    def add_arguments(self, parser):
        parser.add_argument(
            'version_id',
            type=int,
            help='ModelVersion ID to export'
        )
        parser.add_argument(
            '--format',
            type=str,
            default='ctranslate2',
            choices=['ctranslate2', 'ggml'],
            help='Export format (default: ctranslate2)'
        )
        parser.add_argument(
            '--quantization',
            type=str,
            default='int8',
            help='Quantization type (e.g., int8, int16, float16)'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output directory (default: media/exported_models/)'
        )

    def handle(self, *args, **options):
        version_id = options['version_id']
        export_format = options['format']
        quantization = options['quantization']
        output_dir = options.get('output')

        # Get model version
        try:
            version = ModelVersion.objects.get(id=version_id)
        except ModelVersion.DoesNotExist:
            raise CommandError(f'ModelVersion with id={version_id} does not exist')

        self.stdout.write(
            self.style.SUCCESS(
                f'Exporting model version: {version.version}\n'
                f'Format: {export_format}\n'
                f'Quantization: {quantization}'
            )
        )

        # Check if merged model exists
        if not version.merged_model_path or not Path(version.merged_model_path).exists():
            raise CommandError(
                f'Merged model not found at: {version.merged_model_path}\n'
                'Please ensure the training run completed successfully.'
            )

        # Set output directory
        if not output_dir:
            output_dir = Path(settings.MEDIA_ROOT) / 'exported_models' / f'{version.version}'
        else:
            output_dir = Path(output_dir)

        # Export
        exporter = ModelExporter(version.merged_model_path, str(output_dir))

        try:
            if export_format == 'ctranslate2':
                exported_path = exporter.export_ctranslate2(quantization=quantization)
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Exported to CTranslate2: {exported_path}')
                )
            elif export_format == 'ggml':
                exported_path = exporter.export_ggml(quantization=quantization)
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Exported to GGML: {exported_path}')
                )

            # Update model version
            if export_format == 'ctranslate2':
                version.ctranslate2_path = exported_path
            elif export_format == 'ggml':
                version.ggml_path = exported_path

            version.save()

            self.stdout.write(
                self.style.SUCCESS('\n✓ Export complete!')
            )

        except Exception as e:
            raise CommandError(f'Export failed: {e}')
