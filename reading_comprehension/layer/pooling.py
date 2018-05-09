import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["MaxPooling", "AveragePooling"]

class MaxPooling(object):
    """max pooling layer"""
    def __init__(self,
                 scope="maxpool"):
        """initialize max pooling layer"""
        self.scope = scope
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call max pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_mask = tf.expand_dims(input_mask, axis=-1)
            input_pool = tf.reduce_max(input_data * input_mask, axis=-2)
            
            return input_pool

class AveragePooling(object):
    """average pooling layer"""
    def __init__(self,
                 scope="avgpool"):
        """initialize average pooling layer"""
        self.scope = scope
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call average pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_mask = tf.expand_dims(input_mask, axis=-1)
            input_sum = tf.reduce_sum(input_data * input_mask, axis=-2)
            input_count = tf.count_nonzero(input_mask, axis=-2)
            input_pool = 1.0 * input_sum / input_count
            
            return input_pool
