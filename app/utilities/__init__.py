import cv2
import math
import numpy as np

ALPHA = 0.2

ACTIVE_PLAYER_COLORS = {
    "text": (0, 0, 0),
    "background": (0, 215, 255),
}

RAINBOW_COLORS = [
    (255, 0, 127),
    (106, 65, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 127, 255),
    (0, 0, 255),
]

RED_COLOR = (0, 0, 255)

COLORS = {
    "pelota": {
        "text": (0, 0, 0),
        "background": (255, 255, 255),
    },
    "jugador": {
        "text": (255, 255, 255),
        "background": (225, 105, 65),
    },
    "paleta": {
        "text": (255, 255, 255),
        "background": (220, 20, 60),
    },
    "red": {
        "text": (0, 0, 0),
        "background": (178, 190, 181),
    },
    "mesa": {
        "text": (255, 255, 255),
        "background": (34, 139, 34),
    },
}
IMAGES_FOLDER = "images"
MAX_AMOUNTS = {
    "pelota": 1,
    "jugador": 2,
    "paleta": 2,
    "mesa": 1,
    "red": 1,
}
MODEL_PATH = "model/trained_model.pt"
PROCESSED_FILE_SUFFIX = "pong-hawk"
VIDEOS_FOLDER = "videos"


def center(start, end):
    return (
        int((start[0] + end[0]) / 2),
        int((start[1] + end[1]) / 2),
    )


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def is_contained(
    thing: tuple[tuple[int, int], tuple[int, int]],
    container_boundaries: tuple[int, int],
):
    start_x = thing[0][0]
    end_x = thing[1][0]
    return start_x > container_boundaries[0] and end_x < container_boundaries[1]


def image_to_rgb(image):
    return image[..., ::-1]


def to_detection(row):
    detection_name = row["name"]

    return {
        "name": detection_name,
        "confidence": row["confidence"],
        "start": (int(row["xmin"]), int(row["ymin"])),
        "end": (int(row["xmax"]), int(row["ymax"])),
        "colors": COLORS[detection_name],
    }


def draw_label(
    text: str, start_position: tuple[int, int], colors: dict[str, str], frame
):
    label_position = (start_position[0], start_position[1] - 10)
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    label_background_end_position = (
        start_position[0] + width,
        start_position[1] - height * 2,
    )

    # Label
    cv2.rectangle(
        frame, start_position, label_background_end_position, colors["background"], -1
    )
    cv2.putText(
        frame, text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["text"], 2
    )


def draw_detection(image, detection, is_greater_than_max_count=False):
    start_point = detection["start"]
    detection_name = detection["name"]
    confidence = detection["confidence"]
    colors = detection["colors"]

    # Some positioning for the label and background...
    label_text = f"{detection_name} {round(confidence * 100)}%"
    border_color = RED_COLOR if is_greater_than_max_count else colors["background"]

    # Draw a rectangle
    cv2.rectangle(
        image, detection["start"], detection["end"], border_color, thickness=2
    )
    # Draw text in image with background
    draw_label(label_text, start_point, colors, image)


def process_detection(
    model_result,
    last_table_position: tuple[int, int],
    is_upper_player_playing: bool | None = None,
):
    """Main function which process the detections
    of a single frame or image"""
    detections = model_result.pandas().xyxy[0]
    detections_by_type = {object_name: 0 for object_name in MAX_AMOUNTS.keys()}
    balls = []
    players = []
    web = None
    paddles = []
    table = None
    things = []

    # Detections in current frame
    for i in detections.index:
        detection = to_detection(detections.iloc[i])

        object_name = detection["name"]
        detections_by_type[object_name] += 1
        max_count_exceeded = detections_by_type[object_name] > MAX_AMOUNTS[object_name]

        if max_count_exceeded and object_name != "pelota":
            continue

        if object_name == "jugador":
            players.append(detection)
            things.append(detection)
        elif object_name == "mesa":
            table = detection
            things.append(detection)
        elif object_name == "pelota":
            balls.append(detection)
        elif object_name == "red":
            web = detection
            things.append(detection)
        else:
            paddles.append(detection)

    # Area of interest
    x_left_boundary = min(things, key=lambda thing: thing["start"][0], default=None)
    x_right_boundary = max(things, key=lambda thing: thing["end"][0], default=None)

    x_left_boundary = x_left_boundary["start"][0] if x_left_boundary else 0
    x_right_boundary = x_right_boundary["end"][0] if x_right_boundary else 9999

    # Update last_table_position if table was found
    if table is not None:
        last_table_position = center(table["start"], table["end"])

    # Paddles can only exist between cointainer boundaries
    paddles = filter(
        lambda paddle: is_contained(
            (paddle["start"], paddle["end"]), (x_left_boundary, x_right_boundary)
        ),
        paddles,
    )

    # If there where multiple detections of the ball
    # we need to do something about that...
    closest_ball = None
    if len(balls) > 0:
        # Filter out balls out of area of interest
        balls = filter(
            lambda ball: is_contained(
                (ball["start"], ball["end"]), (x_left_boundary, x_right_boundary)
            ),
            balls,
        )
        # Find the closest ball to the table
        closest_ball = min(
            balls,
            key=lambda ball: distance(
                last_table_position, center(ball["start"], ball["end"])
            ),
            default=None,
        )

    # Check if is_upper_player_playing
    if (
        is_upper_player_playing is None
        and closest_ball is not None
        and last_table_position is not None
    ):
        is_upper_player_playing = (
            center(closest_ball["start"], closest_ball["end"])[1]
            < last_table_position[1]
        )

    # Draw player rectangles
    for player in players:
        player_position = center(player["start"], player["end"])
        is_upper_player = player_position[1] < last_table_position[1]
        is_playing = (
            is_upper_player_playing is not None
            and (is_upper_player and is_upper_player_playing)
            or (not is_upper_player and not is_upper_player_playing)
        )
        player["is_playing"] = is_playing
        if is_playing:
            player["colors"] = ACTIVE_PLAYER_COLORS

    return {
        "balls": list(balls),
        "closest_ball": closest_ball,
        "paddles": list(paddles),
        "players": players,
        "table": table,
        "web": web,
        "boundaries": (x_left_boundary, x_right_boundary),
    }


def debug_draw(detections, frame):
    """Draw everything on frame"""
    table = detections["table"]
    if table is not None:
        draw_detection(frame, table)

    web = detections["web"]
    if web is not None:
        draw_detection(frame, web)

    for paddle in detections["paddles"]:
        draw_detection(frame, paddle)

    for player in detections["players"]:
        draw_detection(frame, player)

    ball = detections["closest_ball"]
    if ball is not None:
        draw_detection(frame, ball)

    # Draw boundaries
    left_boundary = detections["boundaries"][0]
    right_boundary = detections["boundaries"][1]
    image_height = frame.shape[0]
    image_width = frame.shape[1]

    # Draw left gray rectangle
    x, y, w, h = 0, 0, left_boundary, image_height
    sub_img = frame[y : y + h, x : x + w]
    black_rectangle = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1 - ALPHA, black_rectangle, ALPHA, 0)
    frame[y : y + h, x : x + w] = res

    # Draw right gray rectangle
    x, y, w, h = right_boundary, 0, image_width - right_boundary, image_height
    sub_img = frame[y : y + h, x : x + w]
    black_rectangle = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1 - ALPHA, black_rectangle, ALPHA, 0)
    frame[y : y + h, x : x + w] = res
