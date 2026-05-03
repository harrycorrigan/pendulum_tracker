"""Main module for pendulum tracker

NOTE: there are lots of magic numbers in this module, i think that's just the reality of computer vision,
      but everything is just colour ranges, blurring, morphing and scaling, these values are arbitrarily
      chosen based on what worked best

"""

import pathlib
import cv2
from cv2.typing import MatLike
import numpy as np
import tqdm

from utils.rect_utils import (
    RectType,
    get_rect,
    get_rect_height,
    get_rect_midpoint,
    get_rect_width,
    offset_rect,
    scale_rect,
    get_largest_contour_from_mask,
    get_largest_rect_from_mask,
)
from utils.misc_utils import (
    create_resize_window,
    cv2_in_range,
    get_angle,
    write_new_csv,
)


def get_protractor_mask_high_res(hsv_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    scale = 0.5
    hsv_image = cv2.resize(
        hsv_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    blurred = cv2.GaussianBlur(hsv_image, (5, 5), 0)

    mask1 = cv2_in_range(blurred, "340°, 40%, 40%", "360°, 70%, 100%")

    kernel = np.ones((50, 50), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

    return cv2.resize(mask1, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)


def get_needle_bounding_box(
    protractor_masked_image: cv2.typing.MatLike,
) -> RectType | tuple[None, None]:
    scale = 0.5
    protractor_masked_image = cv2.resize(
        protractor_masked_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    protractor_masked_image = cv2.GaussianBlur(protractor_masked_image, (49, 49), 0)
    hsv_image = cv2.cvtColor(protractor_masked_image, cv2.COLOR_BGR2HSV)

    # mask to isolate string from protractor background
    mask1 = cv2_in_range(hsv_image, "335°, 20%, 70%", "345°, 30%, 80%")

    kernel = np.ones((50, 50), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((12, 12), np.uint8)
    thickened = cv2.dilate(mask1, kernel, iterations=1)

    try:
        return get_largest_rect_from_mask(thickened)
    except Exception:
        return (None, None)


def get_90deg_marker(protractor_masked_image: MatLike, protractor_rect: RectType):
    # scale whole image down to speed up processing
    scale = 0.5
    protractor_masked_image = cv2.resize(
        protractor_masked_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    protractor_masked_image = cv2.GaussianBlur(protractor_masked_image, (5, 5), 0)
    hsv_image = cv2.cvtColor(protractor_masked_image, cv2.COLOR_BGR2HSV)

    # this is kind of arbitrary, we take the rectangle bounding the protractor and scale it down a bit
    # (setting min height/width to get the rough shape we want)
    # then we offset it because one side of the protractor is
    # larger then the other (note: this is only workable for the two videos, the third doesn't work)
    area_of_interest = scale_rect(
        protractor_rect, 0.1, min_width=150, min_height=700, lin_scale=0.5
    )
    area_of_interest = offset_rect(
        area_of_interest,
        (
            int(get_rect_width(area_of_interest) / 4),
            int(get_rect_height(area_of_interest) / 4),
        ),
    )

    # this gives us a rect _roughly_ at the 90deg marker that we can then filter on colour
    segment_hsv = hsv_image[
        area_of_interest[0][1]:area_of_interest[1][1],
        area_of_interest[0][0]:area_of_interest[1][0],
    ]

    mask = cv2_in_range(segment_hsv, "345°, 63%, 60%", "350°, 75%, 70%")
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    largest_cnt = get_largest_contour_from_mask(mask)

    if largest_cnt is None or cv2.contourArea(largest_cnt) <= 800:
        return None

    x, y, w, h = cv2.boundingRect(largest_cnt)
    return get_rect(
        area_of_interest[0][0] + x, area_of_interest[0][1] + y, w, h, lin_scale=2
    )


def process_video(
    video_path: pathlib.Path, skip_to_s: int = 0, video_scale: float = 1 / 5
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
    if not video_path.exists():
        print("Video at path %s does not exist" % (video_path.absolute()))
        exit()

    cap = cv2.VideoCapture(video_path)

    success, image = cap.read()

    IM_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    IM_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_COUNT = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    SKIP_FRAMES = int(skip_to_s * FPS)

    print("Video dimensions (%s, %s)" % (IM_WIDTH, IM_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, SKIP_FRAMES)

    create_resize_window(
        "window", int(IM_WIDTH * video_scale), int(IM_HEIGHT * video_scale)
    )
    create_resize_window(
        "window2", int(IM_WIDTH * video_scale), int(IM_HEIGHT * video_scale)
    )
    create_resize_window(
        "mask", int(IM_WIDTH * video_scale), int(IM_HEIGHT * video_scale)
    )

    pivot_point = None
    angles: list[tuple[float, float]] = []
    positions: list[tuple[float, float, float]] = []
    frame_count = SKIP_FRAMES
    with tqdm.tqdm(total=FRAME_COUNT - SKIP_FRAMES) as pbar:
        while success:
            # get current time for output data
            curr_time = frame_count / FPS

            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # isolate protractor (filled gaps, string is included in mask)
            mask = get_protractor_mask_high_res(hsv_image)

            # mask the image to get the full color protractor image
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # get the rectangle containing the protractor
            protractor_rect = get_largest_rect_from_mask(mask)
            protractor_rect_midpoint = get_rect_midpoint(protractor_rect)
            protractor_rect_height = get_rect_height(protractor_rect)
            protractor_rect_top = int(
                protractor_rect_midpoint[1] - protractor_rect_height / 2
            )

            # image_midpoint = (int(IM_WIDTH / 2), int(IM_HEIGHT / 4))

            # get the 90 degree marker on the protractor (straight line near the middle of the image)
            new_pivot = get_90deg_marker(masked_image, protractor_rect)

            # when string is covering the 90degree marker it can't be found, only update if we have it
            if new_pivot is not None:
                pivot_point = new_pivot

            # get the string (on the protractor) bounding box
            needle_rect = get_needle_bounding_box(masked_image)

            if needle_rect[0] is None or needle_rect[1] is None:
                print("Failed to find needle position, skipping frame")
                success, image = cap.read()
                continue

            # get the middle of the bounding box, this is our string point, draw it for reference
            needle = get_rect_midpoint(needle_rect)
            cv2.circle(image, needle, 25, (255, 255, 0), thickness=5)

            positions.append((curr_time, needle[0], needle[1]))

            if pivot_point is not None:
                pivot_midpoint = get_rect_midpoint(pivot_point)
                origin = (pivot_midpoint[0], int(protractor_rect_top))

                cv2.rectangle(
                    image, pivot_point[0], pivot_point[1], (0, 255, 0), thickness=10
                )
                cv2.circle(image, origin, 10, (255, 255, 0), thickness=10)
                cv2.line(
                    image,
                    (origin[0], 0),
                    (origin[0], IM_HEIGHT),
                    (255, 0, 0),
                    thickness=10,
                )

                angles.append((curr_time, get_angle(origin, needle)))

            cv2.imshow("window2", image)
            pbar.update(1)
            frame_count += 1

            success, image = cap.read()

            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return angles, positions


INPUT_FILE = "../../high_rest/rec_2.MOV"
OUTPUT_ANGLES = "../output-3-angles.csv"
OUTPUT_POSITIONS = "../output-3-positions.csv"

if __name__ == "__main__":
    video_path = pathlib.Path(INPUT_FILE)
    angles_output_path = pathlib.Path(OUTPUT_ANGLES)
    positions_output_path = pathlib.Path(OUTPUT_POSITIONS)

    angle_data, position_data = process_video(video_path)

    write_new_csv(angles_output_path, angle_data, ["time", "angle"])
    write_new_csv(positions_output_path, position_data, ["time", "x", "y"])
