import codecs
import collections
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["ExtrinsicEvalLog", "DecodingEvalLog", "EvalLogger"]

class ExtrinsicEvalLog(collections.namedtuple("ExtrinsicEvalLog", ("metric", "score", "sample_output", "sample_size"))):
    pass

class DecodingEvalLog(collections.namedtuple("DecodingEvalLog", ("sample_input", "sample_output", "sample_reference"))):
    pass

class EvalLogger(object):
    """evaluation logger"""    
    def __init__(self,
                 output_dir):
        """extrinsic evaluation result"""
        self.extrinsic_eval = None
        """extrinsic evaluation result"""
        self.decoding_eval = None
        
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
    
    def update_decoding_eval(self,
                             eval_result_list):
        """update evaluation logger based on decoding evaluation result"""
        self.decoding_eval = eval_result_list
    
    def check_decoding_eval(self):
        """check decoding evaluation result"""
        sample_size = len(self.decoding_eval)
        for i in range(sample_size):
            eval_result = self.decoding_eval[i]
            log_line = "====================================="
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            log_line = "sample {0} - input: {1}".format(i+1, eval_result.sample_input).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            decode_output = self.decode_sample_output[i]
            log_line = "sample {0} - output: {1}".format(i+1, eval_result.sample_output).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            decode_reference = self.decode_sample_reference[i]
            log_line = "sample {0} - reference: {1}".format(i+1, eval_result.sample_reference).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
