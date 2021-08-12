from argparse import ArgumentParser
import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()

    # load pt file
    pt_file = torch.load(args.file_path)
    print(pt_file)