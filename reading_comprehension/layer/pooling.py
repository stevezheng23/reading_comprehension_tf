import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["MaxPooling", "AveragePooling"]

class MaxPooling(object):
    """max pooling layer"""
    def __init__(self,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="max_pool"):
        """initialize max pooling layer"""
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call max pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            output_pool = tf.reduce_max(input_data * input_mask, axis=-2)
            output_mask = tf.squeeze(tf.reduce_max(input_mask, axis=-2, keep_dims=True), axis=-2)
        
        return output_pool, output_mask

class AveragePooling(object):
    """average pooling layer"""
    def __init__(self,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="avg_pool"):
        """initialize average pooling layer"""
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call average pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_sum = tf.reduce_sum(input_data * input_mask, axis=-2)
            input_count = tf.count_nonzero(input_mask, axis=-2, dtype=tf.float32)
            output_mask = tf.squeeze(tf.reduce_max(input_mask, axis=-2, keep_dims=True), axis=-2)
            output_pool = 1.0 * input_sum / (input_count - output_mask + 1.0)
        
        return output_pool, output_mask
