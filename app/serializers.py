from app.models import Game
from rest_framework import serializers


class GameSerializer(serializers.ModelSerializer):
    class Meta:
        model = Game
        fields = [
            'id',
            'name',
            'video',
            'processed_video',
            'heatmap',
            'status',
            'strategy',
        ]
        read_only_fields = [
            'processed_video',
            'heatmap',
            'status',
            'strategy',
        ]
