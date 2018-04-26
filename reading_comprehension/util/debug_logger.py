import codecs
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["DebugLogger"]

class DebugLogger(object):
    """debug logger"""    
    def __init__(self,
                 output_dir):
        """initialize debug logger"""       
        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)
        self.log_file = os.path.join(output_dir, "debug_{0}.log".format(time.time()))
        self.log_writer = codecs.getwriter("utf-8")(tf.gfile.GFile(self.log_file, mode="a"))
    
    def log_print(self,
                  message):
        """log and print debugging message"""
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        log_line = "{0}: {1}".format(time_stamp, message).encode('utf-8')
        self.log_writer.write("{0}\r\n".format(log_line))
        print(log_line)
