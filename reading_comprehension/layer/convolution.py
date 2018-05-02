import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Conv1D"]

class Conv(object):
    """convolution layer"""
    def __init__(self,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 trainable=True,
                 scope="conv"):
        """initialize convolution layer"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.num_filter = num_filter
            self.window_size = window_size
            self.stride_size = stride_size
            self.padding_type = padding_type
            self.input_type = input_type
            self.trainable = trainable
            
            self.conv_layer = tf.layers.Conv1D(filters=self.num_filter,
                kernel_size=window_size, strides=stride_size, padding=self.padding_type, trainable=trainable)
    
    def __call__(self,
                 input_data):
        """generate convolution layer output"""
        input_conv = self.conv_layer(input_data)
        
        return input_conv

class Conv1D(Conv):
    """1d convolution layer"""
    def __init__(self,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 trainable=True,
                 scope="conv1d"):
        """initialize 1d convolution layer"""
        super(Conv, self).__init__(num_filter=num_filter, window_size=window_size,
            stride_size=stride_size, padding_type=padding_type, trainable=trainable, scope=scope)
    
    def __call__(self,
                 input_data):
        """generate 1d convolution layer output"""
        input_conv = super(Conv, self).__call__(input_data)
        
        return input_conv

class Conv2D(Conv):
    """2d convolution layer"""
    def __init__(self,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 trainable=True,
                 scope="conv2d"):
        """initialize 2d convolution layer"""
        super(Conv, self).__init__(num_filter=num_filter, window_size=window_size,
            stride_size=stride_size, padding_type=padding_type, trainable=trainable, scope=scope)
    
    def __call__(self,
                 input_data):
        """generate 2d convolution layer output"""
        (batch_size, dim1_length, dim2_length,
            input_embed_dim) = tf.shape(input_data)
        input_data = tf.reshape(input_data,
            shape=[batch_size * dim1_length, dim2_length, input_embed_dim])
        input_conv = self.conv_layer(input_data)
        _, dim2_length, input_embed_dim = tf.shape(input_conv)
        input_conv = tf.reshape(input_conv,
            shape=[batch_size, dim1_length, dim2_length, input_embed_dim])
        
        return input_conv
