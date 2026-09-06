import cv2
import numpy as np


# Official badminton doubles court dimensions in metres
COURT_WIDTH = 6.10
COURT_LENGTH = 13.40


def calculate_perspective_transform(source_points):
    """
    Calculate perspective transformation matrix.

    source_points must contain the four court corners
    in this order:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    if len(source_points) != 4:
        raise ValueError(
            "Exactly 4 court corner points are required"
        )

    source_points = np.float32(source_points)

    destination_points = np.float32([
        [0, 0],
        [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_LENGTH],
        [0, COURT_LENGTH]
    ])

    matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    return matrix


def transform_point(point, matrix):
    """
    Transform a pixel coordinate into
    real badminton court coordinates.
    """

    point_array = np.float32([
        [[point[0], point[1]]]
    ])

    transformed_point = cv2.perspectiveTransform(
        point_array,
        matrix
    )

    x = float(transformed_point[0][0][0])
    y = float(transformed_point[0][0][1])

    return {
        "court_x": round(x, 3),
        "court_y": round(y, 3)
    }


def is_point_inside_court(court_x, court_y):
    """
    Check whether a transformed point
    lies inside the badminton court.
    """

    return (
        0 <= court_x <= COURT_WIDTH
        and
        0 <= court_y <= COURT_LENGTH
    )
