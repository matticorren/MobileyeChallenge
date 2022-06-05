# External imports
import os
import json
import numpy as np
from pathlib import Path

# Internal imports
from plot_image_with_labels import plot_image_with_labels


class Handler(object):
    """
    This class implements an object responsible for data restructuring.
    """
    def __init__(self, input_dir, output_dir):
        """
        This function initialize the Handler Object.
        :param input_dir: The dir of the training data
        :param output_dir: The dir to put the output in.
        """
        if os.path.exists(output_dir):
            self.output_dir = output_dir
        else:
            raise InvalidPath(output_dir)
        if os.path.exists(input_dir):
            self.files = {Path(f).stem: os.path.join(input_dir, f) for f in os.listdir(input_dir)}
        else:
            raise InvalidPath(input_dir)
        with open(self.files["data_structure"]) as f:
            self.data_struct = json.load(f)
        self.frames = self._extract_bin("frames")
        self.ids = self._extract_bin("ids")
        self.sticks = self._extract_bin("sticks")
        self.ratios = self._extract_bin("ratios")
        self.images = self._extract_bin("images")
        print(self.images)

    def _extract_bin(self, attribute):
        """
        This function extracts the data of a relevant bin file
        :param attribute: the attribute's bin to be extracted
        :return: a 4D numpy Array.
        """
        attr = self.data_struct[attribute]
        base = np.fromfile(self.files[attribute], attr["type"])
        h, w, c = attr["height"], attr["width"], attr["channels"]
        return base.reshape(base.size//(h*w), h, w, c)

    def validate_extraction(self, index):
        """
        This function validates whether the extraction of the data was successful
        :param index: The index of the frame to be shown.
        """
        plot_image_with_labels(image=self.images[index].reshape(80, 80),
                               sticks=self.sticks[index].reshape(9),
                               sample_id=self.ids[index].reshape(1)[0],
                               sample_frame=self.frames[index].reshape(1)[0],
                               ratios=self.ratios[index].reshape(4))


class InvalidPath(Exception):
    """
    This class implements an error occuring when the user inputs a problematic path.
    """
    def __init__(self, path, message="The path {0} doesn't exist. Be sure to enter a valid one."):
        """
        This class initialize the InvalidPath Error object.
        :param path: The problematic path given by the user.
        :param message: The message to be shown in the console.
        """
        self.path = path
        self.message = message.format(self.path)
        super().__init__(self.message)
