import argparse
import json

import numpy as np
import tensorflow as tf

from util.debug_logger import *
from util.default_util import *
from util.param_util import *
from util.data_util import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def train(logger,
          hyperparams):
    pass

def evaluate(logger,
             hyperparams):
    pass

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train'):
        train(logger, hyperparams)
    elif (args.mode == 'eval'):
        evaluate(logger, hyperparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
