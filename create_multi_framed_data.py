# External imports
import numpy as np
import argparse


def create_multi_framed_data(input, output):
    pass


def main():
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
