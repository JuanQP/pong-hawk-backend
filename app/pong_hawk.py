import cv2
import numpy as np
import torch

from pathlib import Path
from .utilities import (
    MODEL_PATH,
    RAINBOW_COLORS,
    center,
    process_detection,
)

def process_video(video_path: str):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
    model.conf = 0.4

    video = cv2.VideoCapture(video_path)
    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(video.get(cv2.CAP_PROP_FPS), 1)

    SCREEN_CENTER = center((0, 0), (width, height))

    path = Path(video_path)
    new_file_path = path.with_name(f"{path.stem}-processed.webm")
    heatmap_exported_video = cv2.VideoWriter(
        new_file_path.as_posix(),
        cv2.VideoWriter_fourcc(*"VP90"),
        fps,
        (width, height),
    )

    previous_ball_positions = []
    # Default values...
    last_table_position = SCREEN_CENTER
    is_upper_player_playing = None

    # Heatmap matrix
    # This is where we are going to store
    # the ball positions in a grid format
    shape = (width // 20, height // 20)
    heatmap_matrix = np.zeros(shape, dtype=np.uint8)

    # Start processing
    while True:
        continues, frame = video.read()

        if not continues:
            break

        result = model(frame)

        processed_detections = process_detection(
            result, last_table_position, is_upper_player_playing
        )

        table = processed_detections["table"]
        if table is not None:
            last_table_position = center(table["start"], table["end"])

        ball = processed_detections["closest_ball"]
        if ball is not None:
            previous_ball_positions.append(ball)
            is_upper_player_playing = (
                center(ball["start"], ball["end"])[1] < last_table_position[1]
            )
            # Update heatmap matrix
            ball_position = center(ball["start"], ball["end"])
            x, y = round(ball_position[0] / 20), round(ball_position[1] / 20)
            heatmap_matrix[x][y] += 1

        # Draw a rainbow trail
        extra_radius = 0
        for i, ball in enumerate(previous_ball_positions[-7:]):
            # Make more recent dots larger
            extra_radius = i // 3
            dot_color = RAINBOW_COLORS[i]
            cv2.circle(
                frame,
                center(ball["start"], ball["end"]),
                radius=2 + extra_radius,
                color=dot_color,
                thickness=-1,
            )

        # Map histogram to OpenCV 2D grayscale image and resize it
        # to be the same size as video frame
        max_so_far = heatmap_matrix.max()
        heatmap_grayscale = (
            np.array(heatmap_matrix.T / max_so_far * 255, dtype=np.uint8)
            if max_so_far > 0
            else np.zeros((1, 1), dtype=np.uint8)
        )
        resized = cv2.resize(
            heatmap_grayscale, (width, height), interpolation=cv2.INTER_AREA
        )
        gaussian = cv2.GaussianBlur(resized, (15, 15), 0)
        color_heatmap = cv2.applyColorMap(gaussian, cv2.COLORMAP_JET)
        # "Paste" heatmap in last frame
        combined = cv2.addWeighted(frame, 0.5, color_heatmap, 0.5, 0)
        heatmap_exported_video.write(combined)

    # TODO: Create a png with the last frame
    image_path = path.with_name(f"{path.stem}-processed.png")
    cv2.imwrite(image_path.as_posix(), combined)
    video.release()
    heatmap_exported_video.release()

    cv2.destroyAllWindows()
    return new_file_path.name, image_path.name
