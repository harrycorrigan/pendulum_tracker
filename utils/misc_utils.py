"""Miscellaneous utilities.

Contains basic helper functions.

"""

import csv
import pathlib

import numpy as np
import cv2
import math


def google_hsv_to_opencv(hsv_str: str) -> cv2.typing.MatLike:
    """Convert 'google' hsv string to opencv compatible format

    Args:
        hsv_str (str): A HSV string as copied from google colour picker (in the format x° S% V%),

    Returns:
        colour (np.array): The converted HSV string in a numpy array format (x°/2, S% -> S (0-255), V% -> V (0-255))
    """
    hue, saturation, value = hsv_str.split(", ")
    return np.array(
        [
            int(hue.replace("°", "")) / 2,
            (int(saturation.replace("%", "")) / 100) * 255,
            (int(value.replace("%", "")) / 100) * 255,
        ]
    )


def cv2_in_range(
    image: cv2.typing.MatLike, lower_google_hsv: str, upper_google_hsv: str
) -> cv2.typing.MatLike:
    """Perform a filter on a image using cv2.inRange and 'google' hsv strings

    Args:
        image (cv2.typing.MatLike): the image to filter
        lower_google_hsv (str): the lower bound in 'google' hsv format
        upper_google_hsv (str): the upper bound in 'google' hsv format

    Returns:
        mask (cv2.typing.MatLike): the binary mask with every matching pixel set to 255
    """
    return cv2.inRange(
        image,
        google_hsv_to_opencv(lower_google_hsv),
        google_hsv_to_opencv(upper_google_hsv),
    )


def create_resize_window(window_name: str, window_width: int, window_height: int):
    """Create and resize a CV2 window with a constant ratio

    Args:
        window_name (str): the name of the window to create
        window_width (int): the width of the window to create
        window_height (int): the height of the window to create
    """

    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, window_width, window_height)


def get_angle(pivot_pos: tuple[float, float], needle_pos: tuple[float, float]) -> float:
    """Get angle between two points using atan2

    Args:
        pivot_pos (tuple[float, float]): vector, the origin point
        needle_pos (tuple[float, float]): vector, the endpoint

    Returns:
        angle (float): The angle between the two points (in degrees)
    """
    offset = (needle_pos[0] - pivot_pos[0], needle_pos[1] - pivot_pos[1])
    return math.degrees(math.atan2(offset[0], offset[1]))


def write_new_csv(output_path: pathlib.Path, data: list, keys: list[str]):
    """Write data to a new CSV file

    Args:
        output_path (pathlib.Path): the target location
        data (list): the data to write
        keys (list[str]): the keys to append at the top of the file
    """

    with open(output_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(keys)
        for result in data:
            csv_writer.writerow(result)
