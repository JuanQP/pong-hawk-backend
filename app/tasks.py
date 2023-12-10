from app.models import Game
from celery import shared_task
from .pong_hawk import process_video as do_process

@shared_task
def process_video(game_id: str):
    game = Game.objects.get(id=game_id)
    game.status = Game.Status.PROCESSING
    game.save()

    print("Processing game...")
    video_path, image_path = do_process(game.video.path)
    game.processed_video = video_path
    game.heatmap = image_path
    game.status = Game.Status.FINISHED
    game.save()
    print("Done!")
