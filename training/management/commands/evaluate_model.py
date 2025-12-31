"""
Django management command to evaluate a model.
"""
from django.core.management.base import BaseCommand, CommandError
from training.models import ModelVersion
from training.tasks import evaluate_model


class Command(BaseCommand):
    help = 'Evaluate a model version'

    def add_arguments(self, parser):
        parser.add_argument(
            'version_id',
            type=int,
            help='ModelVersion ID to evaluate'
        )

    def handle(self, *args, **options):
        version_id = options['version_id']

        # Get model version
        try:
            version = ModelVersion.objects.get(id=version_id)
        except ModelVersion.DoesNotExist:
            raise CommandError(f'ModelVersion with id={version_id} does not exist')

        self.stdout.write(
            self.style.SUCCESS(f'Evaluating model version: {version.version}')
        )

        # Start evaluation
        task = evaluate_model.delay(version.id)

        self.stdout.write(
            self.style.SUCCESS(
                f'Evaluation started. Task ID: {task.id}'
            )
        )
