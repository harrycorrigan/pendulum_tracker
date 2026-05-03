"""Misc utilities for handling rectangles

Contains primarily basic maths functions,
alongside two functions to help with contours in masks.

"""

import cv2

RectType = tuple[tuple[int, int], tuple[int, int]]


def get_rect(x: int, y: int, width: int, height: int, lin_scale: int = 1) -> RectType:
    """Get a rectangle from x,y height and width

    Args:
        x (int): the x position of the rectangle (top left)
        y (int): the y position of the rectangle (top left)
        width (int): the width of the rectangle
        height (int): the height of the rectangle
        lin_scale (int, optional): Factor to linearly scale the rectangle by (multiply all points by lin_scale),
                                   for correction on scaled images. Defaults to 1.

    Returns:
        RectType: Rectangle described by the x,y,w,h
    """
    return (
        (x * lin_scale, y * lin_scale),
        ((x + width) * lin_scale, (y + height) * lin_scale),
    )


def get_rect_midpoint(rect: RectType) -> tuple[int, int]:
    """Get the midpoint of a rectangle

    Args:
        rect (RectType): the rectangle

    Returns:
       int: the midpoint
    """
    x1, y1 = rect[0]
    x2, y2 = rect[1]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_rect_height(rect: RectType) -> int:
    """Get the height of a rectangle

    Args:
        rect (RectType): the rectangle to get the height of

    Returns:
        int: the height
    """
    _, y1 = rect[0]
    _, y2 = rect[1]
    dims = [y1, y2]
    return max(dims) - min(dims)


def get_rect_width(rect: RectType):
    """Get the width of a rectangle

    Args:
        rect (RectType): the rectangle to get the width of

    Returns:
        int: the width
    """
    x1, _ = rect[0]
    x2, _ = rect[1]
    dims = [x1, x2]
    return max(dims) - min(dims)


def offset_rect(rect: RectType, offset: tuple[int, int]) -> RectType:
    """Add a given offset to all points of a rectangle

    Args:
        rect (RectType): the rectangle
        offset (tuple[int, int]): the offset to apply

    Returns:
        RectType: the new, offset, rectangle
    """
    return (
        (rect[0][0] + offset[0], rect[0][1] + offset[1]),
        (rect[1][0] + offset[0], rect[1][1] + offset[1]),
    )


def scale_rect(
    rect: RectType,
    s: float,
    min_width: int = 0,
    min_height: int = 0,
    lin_scale: float = 1,
) -> RectType:
    """Scale a rectangle & apply minimum height/width

    Args:
        rect (RectType): the rectangle to scale
        s (float): the factor to scale by
        min_width (int, optional): the minimum width of the returned rectangle. Defaults to 0.
        min_height (int, optional): the minimum height of the returned rectangle. Defaults to 0.
        lin_scale (int, optional): Factor to linearly scale the rectangle by (multiply all points by lin_scale),
                                   for correction on scaled images. Defaults to 1.

    Returns:
        RectType: the scaled rectangle
    """
    cx, cy = get_rect_midpoint(rect)

    curr_w = get_rect_width(rect)
    curr_h = get_rect_height(rect)

    new_w = max(curr_w * s, min_width)
    new_h = max(curr_h * s, min_height)

    half_w = int(new_w / 2)
    half_h = int(new_h / 2)

    new_p1 = (int((cx - half_w) * lin_scale), int((cy - half_h) * lin_scale))
    new_p2 = (int((cx + half_w) * lin_scale), int((cy + half_h) * lin_scale))

    return new_p1, new_p2


def get_largest_contour_from_mask(
    mask: cv2.typing.MatLike,
) -> cv2.typing.MatLike | None:
    """Get the largest contour (edge) in a binary mask

    Args:
        mask (cv2.typing.MatLike): the mask (binary, 0 or 255) to find the contour in

    Returns:
        cv2.typing.MatLike | None: the largest contour, None if contour cannot be found
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)


def get_largest_rect_from_mask(mask: cv2.typing.MatLike) -> RectType:
    """Get the largest rectangle (contour bounding rect) in a binary mask

    Args:
        mask (cv2.typing.MatLike): the mask (binary, 0 or 255) to find the contour in

    Raises:
        Exception: Raised when no contour is found in the image

    Returns:
        RectType: the rectangle bounding the largest contour
    """

    largest_cnt = get_largest_contour_from_mask(mask)

    if largest_cnt is None:
        raise Exception("Can't get largest rect, no contours")

    x, y, w, h = cv2.boundingRect(largest_cnt)
    p1, p2 = get_rect(x, y, w, h)

    return p1, p2


# this function serves no purpose anymore but it's too beautiful to delete so.

# def get_overlap_gap_rect(r1: RectType, r2: RectType) -> RectType:
#     """Get the overlap (or gap) between two rectangles as another rectangle.
#
#     Args:
#         rect1 (RectType): the first rectangle
#         rect2 (RectType): the second rectangle
#     """
#
#     # NOTE: "highest" visually is actually lowest y value
#     get_highest_rect_point: Callable[[RectType], tuple[int, int]] = lambda r1: r1[0] if r1[0][1] < r1[1][1] else r1[1]
#     get_lowest_rect_point: Callable[[RectType], tuple[int, int]] = lambda r1: r1[0] if r1[0][1] > r1[1][1] else r1[1]
#
#     get_leftmost_rect_point: Callable[[RectType], tuple[int, int]] = lambda r1: r1[0] if r1[0][0] < r1[1][0] else r1[1]
#     get_rightmost_rect_point: Callable[[RectType], tuple[int, int]] = lambda r1: r1[0] if r1[0][0] > r1[1][0] else r1[1]
#
#     def get_highest_rect(rects: list[RectType]) -> RectType:
#         heights = [get_lowest_rect_point(r)[1] for r in rects]
#         return rects[heights.index(min(heights))]
#
#     def get_lowest_rect(rects: list[RectType]) -> RectType:
#         heights = [get_highest_rect_point(r)[1] for r in rects]
#         return rects[heights.index(max(heights))]
#
#     def get_rightmost_rect(rects: list[RectType]):
#         rights = [get_rightmost_rect_point(r)[0] for r in rects]
#         return rects[rights.index(max(rights))]
#
#     def get_leftmost_rect(rects: list[RectType]):
#         lefts = [get_leftmost_rect_point(r)[0] for r in rects]
#         return rects[lefts.index(min(lefts))]
#
#     y1 = get_highest_rect_point(get_lowest_rect([r1, r2]))[1]
#     y2 = get_lowest_rect_point(get_highest_rect([r1, r2]))[1]
#
#     x1 = get_leftmost_rect_point(get_rightmost_rect([r1, r2]))[0]
#     x2 = get_rightmost_rect_point(get_leftmost_rect([r1, r2]))[0]
#
#     return ((x1, y1), (x2, y2))
