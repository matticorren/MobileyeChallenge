import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional, Tuple
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

BACK_FACE_COLOR = 'blue'
FRONT_FACE_COLOR = 'red'
LEFT_FACE_COLOR = 'green'
RIGHT_FACE_COLOR = 'yellow'
FACES_COLORS = np.array([BACK_FACE_COLOR, FRONT_FACE_COLOR, LEFT_FACE_COLOR, RIGHT_FACE_COLOR])

BACK_INDEX_IN_RATIOS = 0
FRONT_INDEX_IN_RATIOS = 1
LEFT_INDEX_IN_RATIOS = 2
RIGHT_INDEX_IN_RATIOS = 3

STICK1_X = 0
STICK1_Y_BOTTOM = 1
STICK1_Y_TOP = 2
STICK2_X = 3
STICK2_Y_BOTTOM = 4
STICK2_Y_TOP = 5
STICK3_X = 6
STICK3_Y_BOTTOM = 7
STICK3_Y_TOP = 8


def convert_sticks_coordinates_system(sticks: np.ndarray, image_size: int) -> np.ndarray:
    """Convert the sticks from a coordinate-system where the origin is at the center and plus/minus 1 at the edges,
    to a one where the origin is at the bottom left and each pixel is a unit.

    Args:
        sticks: in format of [stick1_x, stick1_y_bottom, stick1_y_top,
                              stick2_x, stick2_y_bottom, stick2_y_top,
                              stick3_x, stick3_y_bottom, stick3_y_top]
        image_size: The size of the image, in pixels.

    Returns:
        Converted sticks in the same format.
    """
    half_image_size = image_size / 2
    return half_image_size + half_image_size * sticks


def create_faces_polygons(sticks: np.ndarray, colors: Optional[Tuple[str, str]] = None) -> Tuple[Polygon, Polygon]:
    """Creates two polygons representing the leftmost and rightmost faces.

    Args:
        sticks: in format of [stick1_x, stick1_y_bottom, stick1_y_top,
                              stick2_x, stick2_y_bottom, stick2_y_top,
                              stick3_x, stick3_y_bottom, stick3_y_top]
        colors: the color of left face and the color of right face.

    Returns:
        Two polygons representing the leftmost and rightmost faces.
    """

    leftmost_polygon_kwargs = dict(fill=False, color='purple') if colors is None else dict(alpha=0.3, color=colors[0])
    rightmost_polygon_kwargs = dict(fill=False, color='purple') if colors is None else dict(alpha=0.3, color=colors[1])

    left_face_polygon = Polygon(
        xy=[[sticks[STICK1_X], sticks[STICK1_Y_TOP]],
            [sticks[STICK1_X], sticks[STICK1_Y_BOTTOM]],
            [sticks[STICK2_X], sticks[STICK2_Y_BOTTOM]],
            [sticks[STICK2_X], sticks[STICK2_Y_TOP]]], 
        **leftmost_polygon_kwargs)

    right_face_polygon = Polygon(
        xy=[[sticks[STICK2_X], sticks[STICK2_Y_TOP]],
            [sticks[STICK2_X], sticks[STICK2_Y_BOTTOM]],
            [sticks[STICK3_X], sticks[STICK3_Y_BOTTOM]],
            [sticks[STICK3_X], sticks[STICK3_Y_TOP]]], 
        **rightmost_polygon_kwargs)

    return left_face_polygon, right_face_polygon


def define_faces_colors_by_orientation(ratios: np.ndarray) -> Tuple[str, str]:
    """Defines the color of the patches that cover the vehicle's faces.

    The colors are suited to each of the vehicle faces as follows:
    [back, front, left, right] = [blue, red, green, yellow].

    Args:
        ratios: The fraction of the face width from the total width of the vehicle,
                for each face in the order [back, front, left, right]

    Returns:
        The color of the leftmost face and the color of the rightmost face.
    """
    faces_which_are_non_zero = np.where(ratios > 0)[0]

    if len(faces_which_are_non_zero) == 1:
        return (FACES_COLORS[faces_which_are_non_zero][0],) * 2

    patches_colors = FACES_COLORS[faces_which_are_non_zero]

    # if the orientation is left-back or right-front then the colors are listed in a wrong order
    # (because back comes before left and front comes before right in the ratios array)
    if ({LEFT_INDEX_IN_RATIOS, BACK_INDEX_IN_RATIOS}.issubset(set(faces_which_are_non_zero)) or
            {FRONT_INDEX_IN_RATIOS, RIGHT_INDEX_IN_RATIOS}.issubset(set(faces_which_are_non_zero))):
        patches_colors = patches_colors[::-1]

    return patches_colors


def create_basic_image(image: np.ndarray) -> plt.Axes:
    """Plot the image.

    Args:
        image: a NumPy array holding the image.

    Returns:
        plt.Axes: pyplot axes
    """
    ax = plt.imshow(image, origin='lower', cmap='gray')
    plt.axis('off')
    return ax.axes


def draw_faces(ax: plt.Axes, sticks: np.ndarray, ratios: Optional[np.ndarray] = None):
    """Adds colors to the different vehicle's faces.
    More on that in define_faces_colors_by_orientation function.

    Args:
        ax (plt.Axes): The ax to draw on.
        sticks: in format of [stick1_x, stick1_y_bottom, stick1_y_top,
                              stick2_x, stick2_y_bottom, stick2_y_top,
                              stick3_x, stick3_y_bottom, stick3_y_top]
        ratios: The fraction of the face width from the total width of the vehicle,
                for each face in the order [back, front, left, right]
    """
    colors = define_faces_colors_by_orientation(ratios) if ratios is not None else None
    left_face_polygon, right_face_polygon = create_faces_polygons(sticks, colors)
    ax.add_patch(left_face_polygon)
    ax.add_patch(right_face_polygon)


def add_sample_frame_and_id(sample_frame: int, sample_id: int):
    """Plots the sample's frame and id on the figure.

    Args:
        sample_frame: The frame from which the image was taken from.
        sample_id: The ID of the vehicle in the image.
    """
    frame_and_id = f"frame: {sample_frame}, id: {sample_id}"
    plt.figtext(0.5, 0.01, frame_and_id, ha="center", fontsize=10,
                bbox={"color": "orange", "alpha": 0.5, "pad": 5})


def plot_image_with_labels(image: np.ndarray,
                           sticks: np.ndarray, 
                           sample_id: int, 
                           sample_frame: int,
                           ratios: Optional[np.ndarray] = None):
    """Shows the image of the vehicle with the box drawn on it. 
    When ratios are given, each face is colored according to its type:
        back - red
        front - green
        left - yellow
        right - blue
    When ratios are not given, all faces are colored the same.

    Args:
        image: array of shape 80x80 
        sticks: in format of [stick1_x, stick1_y_bottom, stick1_y_top,
                              stick2_x, stick2_y_bottom, stick2_y_top,
                              stick3_x, stick3_y_bottom, stick3_y_top]
        sample_id: The ID of the vehicle in the image.
        sample_frame: The frame from which the image was taken from.
        ratios: The fraction of the face width from the total width of the vehicle,
                for each face in the order [back, front, left, right]
    """
    plt.figure()
    ax = create_basic_image(image)
    sticks = convert_sticks_coordinates_system(sticks, image_size=image.shape[0])
    draw_faces(ax, sticks, ratios)
    add_sample_frame_and_id(sample_frame, sample_id)
    plt.show()
