# External imports
import argparse

# Internal imports
from handler import Handler

INPUT = ".\\training_data"
OUTPUT = ".\\output"


def create_multi_framed_data(input_dir=INPUT, output_dir=OUTPUT):
    """
    This function reorganizes the data sets as multi-framed
    :param input: the dir containing the original data sets.
    :param output: the dir to save the new data sets.
    """
    handler = Handler(input_dir, output_dir)
    handler.reorganize_dataset()


def main():
    """
    main function
    """
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--single_frame_dir",
                        dest="input",
                        help="Single-Frame dir's path.")
    parser.add_argument("--output_dir",
                        dest="output",
                        help="A dir to save in it the output.")
    args = parser.parse_args()
    create_multi_framed_data(args.input, args.output)


if __name__ == "__main__":
    main()
