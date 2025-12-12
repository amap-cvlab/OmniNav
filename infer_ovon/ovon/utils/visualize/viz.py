from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)


def append_text_to_image(image: np.ndarray, text: List[str], font_size: float = 0.5):
    r"""Appends lines of text on top of an image. First this will render to the
    left-hand side of the image, once that column is full, it will render to
    the right hand-side of the image.
    :param image: the image to put text underneath
    :param text: The list of strings which will be rendered, separated by new lines.
    :returns: A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    left_aligned = True
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y > h:
            left_aligned = False
            y = textsize[1] + 10

        if left_aligned:
            x = 10
        else:
            x = w - (textsize[0] + 10)

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness * 2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return np.clip(image, 0, 255)


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    r"""Flattens nested dict.

    Source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    :param d: Nested dict.
    :param parent_key: Parameter to set parent dict key.
    :param sep: Nested keys separator.
    :return: Flattened dict.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def overlay_text_to_image(image: np.ndarray, text: List[str], font_size: float = 0.5):
    r"""Overlays lines of text on top of an image.

    First this will render to the left-hand side of the image, once that column is full,
    it will render to the right hand-side of the image.

    :param image: The image to put text on top.
    :param text: The list of strings which will be rendered (separated by new lines).
    :param font_size: Font size.
    :return: A new image with text overlaid on top.
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    left_aligned = True
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y > h:
            left_aligned = False
            y = textsize[1] + 10

        if left_aligned:
            x = 10
        else:
            x = w - (textsize[0] + 10)

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness * 2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return np.clip(image, 0, 255)


def overlay_frame(
    frame: np.ndarray, info: Dict[str, Any], additional: Optional[List[str]] = None
) -> np.ndarray:
    """
    Renders text from the `info` dictionary to the `frame` image.
    """

    lines = []
    flattened_info = flatten_dict(info)
    for k, v in flattened_info.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}")
        else:
            try:
                lines.append(f"{k}: {v:.2f}")
            except TypeError:
                pass
    if additional is not None:
        lines.extend(additional)

    frame = overlay_text_to_image(frame, lines, font_size=0.25)

    return frame
