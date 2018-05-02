import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["MaxPooling", "AveragePooling", "Pooling"]

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
            input_sum = tf.reduce_sum(input_data * input_mask, axis=-2)
            input_count = tf.count_nonzero(input_mask, axis=-2)
            input_pool = 1.0 * input_sum / input_count
            
            return input_pool

class Pooling(object):
    """pooling layer"""
    def __init__(self,
                 pooling_type="max",
                 scope="pool"):
        """initialize pooling layer"""
        self.pooling_type = pooling_type
        self.scope = scope
        
        if self.pooling_type == "max":
            self.pooling_layer = MaxPooling(self.scope)
        elif self.pooling_type == "avg":
            self.pooling_layer = AveragePooling(self.scope)
        else:
            raise ValueError("unsupported pooling type {0}".format(self.pooling_type))
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call pooling layer"""
        input_pool = self.pooling_layer(input_data, input_mask)
        
        return input_pool
