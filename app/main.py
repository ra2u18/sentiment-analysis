import argparse
import hashlib
import pathlib
import sys
from typing import List
import numpy as np

from preprocess_train_simple import preprocess_train
from preprocess_test_simple import preprocess_test
from constants import DATA

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [DATASET]...",
        description="Train or test an ML model, goal: classify real and fake tweets"
    )

    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )

    parser.add_argument('--preptrain', type=str2bool, nargs="?", const=True, default=False,
                help='Preprocess the train dataset')

    parser.add_argument('--preptest', type=str2bool, nargs="?", const=True, default=False,
                help='Preprocess the test dataset')

    return parser

def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    if args.preptrain:
        preprocess_train()
    if args.preptest:
        preprocess_test()

if __name__ == '__main__':
    main()