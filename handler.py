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
        self.output_dir = output_dir
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

        :param attribute:
        :return:
        """
        return np.fromfile(self.files[attribute], self.data_struct[attribute]["type"])


class InvalidPath(Exception):
    """

    """
    def __init__(self, path, message="The path {0} doesn't exist. Be sure to enter a valid one."):
        self.path = path
        self.message = message.format(self.path)
        super().__init__(self.message)
