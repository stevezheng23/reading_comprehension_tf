import codecs
import collections
import os.path
import time
import json

import numpy as np
import tensorflow as tf

__all__ = ["BasicInfoEvalLog", "ExtrinsicEvalLog", "DecodingEvalLog", "EvalLogger"]

class BasicInfoEvalLog(collections.namedtuple("BasicInfoEvalLog", ("epoch", "global_step"))):
    pass

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
        self.extrinsic_eval_info = None
        self.extrinsic_eval_detail = None
        self.extrinsic_eval_detail_info = None
        
        """extrinsic evaluation result"""
        self.decoding_eval = None
        self.decoding_eval_info = None
        
        """initialize evaluation logger"""        
        self.output_dir = output_dir
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)
        self.log_file = os.path.join(self.output_dir, "eval_{0}.log".format(time.time()))
        self.log_writer = codecs.getwriter("utf-8")(tf.gfile.GFile(self.log_file, mode="a"))
    
    def update_extrinsic_eval(self,
                              eval_result_list,
                              basic_info):
        """update evaluation logger with extrinsic evaluation result"""
        self.extrinsic_eval = eval_result_list
        self.extrinsic_eval_info = basic_info
    
    def update_extrinsic_eval_detail(self,
                                     eval_result_detail,
                                     basic_info):
        """update evaluation logger with extrinsic evaluation result detail"""
        self.extrinsic_eval_detail = eval_result_detail
        self.extrinsic_eval_detail_info = basic_info
    
    def check_extrinsic_eval(self):
        """check extrinsic evaluation result"""
        for eval_result in self.extrinsic_eval:
            log_line = "epoch={0}, global step={1}, {2}={3}, sample size={4}".format(self.extrinsic_eval_info.epoch,
                self.extrinsic_eval_info.global_step, eval_result.metric, eval_result.score, eval_result.sample_size).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
    
    def check_extrinsic_eval_detail(self):
        """check extrinsic evaluation detail result"""
        eval_detail_file = os.path.join(self.output_dir, "eval_{0}_{1}_{2}.detail".format(self.extrinsic_eval_detail_info.epoch,
            self.extrinsic_eval_detail_info.global_step, time.time()))
        with codecs.getwriter("utf-8")(tf.gfile.GFile(eval_detail_file, mode="w")) as eval_detail_writer:
            if self.extrinsic_eval_detail is None:
                return
            sample_output = json.dumps(self.extrinsic_eval_detail.sample_output, indent=4)
            eval_detail_writer.write(sample_output)
    
    def update_decoding_eval(self,
                             eval_result_list,
                             basic_info):
        """update evaluation logger with decoding evaluation result"""
        self.decoding_eval = eval_result_list
        self.decoding_eval_info = basic_info
    
    def check_decoding_eval(self):
        """check decoding evaluation result"""
        sample_size = len(self.decoding_eval)
        log_line = "epoch={0}, global step={1}, sample size={2}".format(self.decoding_eval_info.epoch,
            self.decoding_eval_info.global_step, sample_size).encode('utf-8')
        self.log_writer.write("{0}\r\n".format(log_line))
        print(log_line)
        
        for i in range(sample_size):
            eval_result = self.decoding_eval[i]
            log_line = "====================================="
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            log_line = "sample {0} - input: {1}".format(i+1, eval_result.sample_input).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            log_line = "sample {0} - output: {1}".format(i+1, eval_result.sample_output).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            log_line = "sample {0} - reference: {1}".format(i+1, eval_result.sample_reference).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
