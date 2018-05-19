import codecs
import collections
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["IntrinsicEvalLog", "ExtrinsicEvalLog", "EvalLogger"]

class IntrinsicEvalLog(collections.namedtuple("IntrinsicEvalLog", ("metric", "score", "sample_size"))):
    pass

class ExtrinsicEvalLog(collections.namedtuple("ExtrinsicEvalLog", ("metric", "score", "sample_output", "sample_size"))):
    pass

class EvalLogger(object):
    """evaluation logger"""    
    def __init__(self,
                 output_dir):
        """initialize evaluation logger"""        
        self.output_dir = output_dir
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)
        self.log_file = os.path.join(self.output_dir, "eval_{0}.log".format(time.time()))
        self.log_writer = codecs.getwriter("utf-8")(tf.gfile.GFile(self.log_file, mode="a"))
