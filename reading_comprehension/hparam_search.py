import argparse

import numpy as np
import tensorflow as tf

from util.param_util import *

def add_arguments(parser):
    parser.add_argument("--base-config", help="path to base config", required=True)
    parser.add_argument("--search-config", help="path to search config", required=True)
    parser.add_argument("--num-group", help="num of hyperparam group", type=int, required=True)
    parser.add_argument("--random-seed", help="random seed", type=int, required=True)
    parser.add_argument("--output-dir", help="path to output dir", required=True)

def main(args):
    hyperparams = load_hyperparams(args.base_config)
    hyperparams_group = search_hyperparams(hyperparams,
        args.search_config, args.num_group, args.random_seed)
    create_hyperparams_file(hyperparams_group, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
