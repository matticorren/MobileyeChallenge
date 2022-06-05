# External imports
import pytest

# Internal imports
from plot_image_with_labels import plot_image_with_labels
from handler import Handler

# Constants
INPUT = ".\\training_data"
OUTPUT = ".\\output"
TEST_SINGLE = ".\\tests\\single"
TEST_MULTIPLE = ".\\tests\\single"

handler1 = Handler(input_dir=INPUT, output_dir=TEST_SINGLE)
handler2 = Handler(input_dir=OUTPUT, output_dir=TEST_MULTIPLE)


@pytest.mark.parametrize("index", "handler", [121, handler1], [151, handler1], [188, handler1])
def validate_extraction_single_frame(index, handler):
    """
    This function validates whether the extraction of the data was successful.
    :param index: The index of the frame to be shown.
    :param handler: the Handler object.
    """
    plot_image_with_labels(image=handler.images[index].reshape(80, 80),
                           sticks=handler.sticks[index].reshape(9),
                           sample_id=handler.ids[index].reshape(1)[0],
                           sample_frame=handler.frames[index].reshape(1)[0],
                           ratios=handler.ratios[index].reshape(4))


@pytest.mark.parametrize("index", "handler", [120, handler2], [151, handler2], [188, handler2])
def validate_extraction_both_frames(index, handler):
    """
    This function validates whether the manipulation of the data was successful.
    :param index: The index of the frame to be shown.
    :param handler: the Handler object.
    """
    for i in range(2):
        plot_image_with_labels(image=handler.images[index][:, :, i].reshape(80, 80),
                               sticks=handler.sticks[index][:, :, i].reshape(9),
                               sample_id=handler.ids[index].reshape(1)[0],
                               sample_frame=handler.frames[index][:, :, i].reshape(1)[0],
                               ratios=handler.ratios[index][:, :, i].reshape(4))


def test():
    validate_extraction_single_frame(120, handler1)
    validate_extraction_both_frames(151, handler2)
    validate_extraction_both_frames(188, handler2)


if __name__ == "__main__":
    test()
