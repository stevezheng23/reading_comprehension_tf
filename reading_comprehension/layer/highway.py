import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Highway"]

class Highway(object):
    """highway network layer"""
    def __init__(self,
                 trainable=True,
                 scope="highway"):
        """initialize highway network layer"""
        self.trainable = trainable
        self.scope = scope
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call highway network layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_highway = input_data * input_mask
            
            return input_highway
