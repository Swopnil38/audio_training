"""
Django management command to start training from CLI.
"""
from django.core.management.base import BaseCommand, CommandError
from training.models import TrainingConfig, TrainingRun
from training.tasks import run_training_pipeline


class Command(BaseCommand):
    help = 'Start a training run'

    def add_arguments(self, parser):
        parser.add_argument(
            'config_id',
            type=int,
            help='TrainingConfig ID to use'
        )
        parser.add_argument(
            '--sync',
            action='store_true',
            help='Run training synchronously (blocking) instead of async'
        )

    def handle(self, *args, **options):
        config_id = options['config_id']
        sync = options['sync']

        # Get config
        try:
            config = TrainingConfig.objects.get(id=config_id)
        except TrainingConfig.DoesNotExist:
            raise CommandError(f'TrainingConfig with id={config_id} does not exist')

        self.stdout.write(
            self.style.SUCCESS(f'Starting training with config: {config.name}')
        )

        # Create training run
        run = TrainingRun.objects.create(config=config)

        self.stdout.write(f'Created TrainingRun with id={run.id}')

        # Start training
        if sync:
            self.stdout.write(
                self.style.WARNING('Running training synchronously (this may take a while)...')
            )
            result = run_training_pipeline(run.id)
            self.stdout.write(
                self.style.SUCCESS(f'Training completed: {result}')
            )
        else:
            task = run_training_pipeline.delay(run.id)
            run.celery_task_id = task.id
            run.save()

            self.stdout.write(
                self.style.SUCCESS(
                    f'Training started asynchronously. Task ID: {task.id}\n'
                    f'Monitor progress at: /api/training/runs/{run.id}/'
                )
            )
