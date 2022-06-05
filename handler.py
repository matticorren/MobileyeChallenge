# External imports
import os
import json
import numpy as np
from pathlib import Path


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

    def get_all_previous_and_current_frames_idx(self, k=1):
        """
        This function clusters together frames based on the k number representing the number of
        previous frames.
        :param k: the number of previous frames to make use of.
        :return: a numpy Array of all the relevant frames clustered together.
        """
        return np.concatenate([np.repeat(np.flatnonzero(self.ids == i), k+1)[1:-1].reshape(-1, k+1)
                               for i in np.unique(self.ids)])

    def cluster_data(self, attribute):
        """
        This function clusters a data set elements together as previous and current elements.
        :param attribute: The specific dataset to be clustered.
        :return: numpy Array of the clustered data.
        """
        idx = self.get_all_previous_and_current_frames_idx()
        attr = self.__getattribute__(attribute)
        prev = attr[idx[:, 0]]
        curr = attr[idx[:, 1]]
        return np.concatenate((curr, prev), axis=3)

    def reorganize_dataset(self):
        """
        This function is responsible for the reorganization of the datasets according
        to the instructions.
        """
        self.cluster_data("images").tofile(os.path.join(self.output_dir, "images.bin"))
        self.cluster_data("sticks").tofile(os.path.join(self.output_dir, "sticks.bin"))
        self.cluster_data("frames").tofile(os.path.join(self.output_dir, "frames.bin"))
        self.cluster_data("ratios").tofile(os.path.join(self.output_dir, "ratios.bin"))


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
