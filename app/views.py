from app import tasks
from app.models import Game
from app.serializers import GameSerializer
from rest_framework import viewsets
from rest_framework.response import Response


class HiViewSet(viewsets.ViewSet):
    """
    Pong Hawk Welcome
    """
    def list(self, request):
        return Response(
            {"message": "Hi! Looks like everything is working as expected 🙂"}
        )


class GameViewSet(viewsets.ModelViewSet):
    serializer_class = GameSerializer
    queryset = Game.objects.all()

    def perform_create(self, serializer):
        new_game = serializer.save()
        tasks.process_video.apply_async(args=[new_game.id], queue="videos")
