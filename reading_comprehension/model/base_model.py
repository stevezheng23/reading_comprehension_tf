import collections
import os.path

import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["TrainResult", "EvaluateResult", "InferResult", "BaseModel"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class EvaluateResult(collections.namedtuple("EvaluateResult", ("loss", "batch_size", "word_count"))):
    pass

class InferResult(collections.namedtuple("InferResult",
    ("logits", "sample_id", "sample_word", "sample_sentence", "batch_size", "summary"))):
    pass

class BaseModel(object):
    """reading comprehension base model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="base"):
        """initialize mrc base model"""
        self.logger = logger
        self.hyperparams = hyperparams
        self.data_pipeline = data_pipeline
        self.mode = mode
        self.scope = scope
        
        self.num_gpus = self.hyperparams.device_num_gpus
        self.default_gpu_id = self.hyperparams.device_default_gpu_id
        self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
            .format(self.num_gpus, self.default_gpu_id))
