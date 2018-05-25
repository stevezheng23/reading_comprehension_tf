import codecs
import collections
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["ExtrinsicEvalLog", "EvalLogger"]

class ExtrinsicEvalLog(collections.namedtuple("ExtrinsicEvalLog", ("metric", "score", "sample_output", "sample_size"))):
    pass

class EvalLogger(object):
    """evaluation logger"""    
    def __init__(self,
                 output_dir):
        """extrinsic evaluation result"""
        self.extrinsic_eval = None
        
        """initialize evaluation logger"""        
        self.output_dir = output_dir
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)
        self.log_file = os.path.join(self.output_dir, "eval_{0}.log".format(time.time()))
        self.log_writer = codecs.getwriter("utf-8")(tf.gfile.GFile(self.log_file, mode="a"))
    
    def update_extrinsic_eval(self,
                              eval_result_list):
        """update evaluation logger based on extrinsic evaluation result"""
        self.extrinsic_eval = eval_result_list
    
    def check_extrinsic_eval(self):
        """check extrinsic evaluation result"""
        for eval_result in self.extrinsic_eval:
            log_line = "{0}={1}, sample size={2}".format(eval_result.metric,
                eval_result.score, eval_result.sample_size).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
