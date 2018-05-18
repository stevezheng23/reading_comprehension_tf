import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Pointer"]

class Pointer(object):
    """pointer layer"""
    def __init__(self,
                 trainable=True,
                 scope="pointer"):
        """initialize pointer layer"""
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.transform_layer = tf.layers.Dense(units=1, activation=None, trainable=trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call pointer layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_transform = self.transform_layer(input_data)
            output_pointer = tf.nn.softmax(input_transform, dim=1)
            output_pointer = tf.expand_dims(tf.argmax(output_pointer, axis=-2), axis=-1)
            output_mask = tf.reduce_max(input_mask, axis=-2, keep_dims=True)
        
        return output_pointer, output_mask
