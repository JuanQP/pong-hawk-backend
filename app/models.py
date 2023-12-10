from django_cleanup import cleanup
from django.db import models

# Create your models here.
@cleanup.select
class Game(models.Model):
    """Video container"""
    class Status(models.TextChoices):
        FINISHED = "FI", "Finished"
        PENDING = "PE", "Pending"
        PROCESSING = "PR", "Processing"
    name = models.TextField()
    video = models.FileField()
    processed_video = models.FileField(null=True)
    heatmap = models.FileField(null=True)
    status = models.CharField(
        max_length=2,
        choices=Status.choices,
        default=Status.PENDING,
    )
