# External imports
import pytest

# Internal imports
from plot_image_with_labels import plot_image_with_labels


def validate_extraction_single_frame(index):
    """
    This function validates whether the extraction of the data was successful
    :param index: The index of the frame to be shown.
    """
    plot_image_with_labels(image=self.images[index].reshape(80, 80),
                           sticks=self.sticks[index].reshape(9),
                           sample_id=self.ids[index].reshape(1)[0],
                           sample_frame=self.frames[index].reshape(1)[0],
                           ratios=self.ratios[index].reshape(4))


def validate_extraction_both_frames(images, sticks, frames, ratios, index):
    """
    This function validates whether the extraction of the data was successful
    :param index: The index of the frame to be shown.
    """
    plot_image_with_labels(image=images[index][:, :, 0].reshape(80, 80),
                           sticks=sticks[index][:, :, 0].reshape(9),
                           sample_id=self.ids[index].reshape(1)[0],
                           sample_frame=frames[index][:, :, 0].reshape(1)[0],
                           ratios=ratios[index][:, :, 0].reshape(4))
    plot_image_with_labels(image=images[index][:, :, 1].reshape(80, 80),
                           sticks=sticks[index][:, :, 1].reshape(9),
                           sample_id=self.ids[index].reshape(1)[0],
                           sample_frame=frames[index][:, :, 1].reshape(1)[0],
                           ratios=ratios[index][:, :, 1].reshape(4))