import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Conv1D"]

class Conv1D(object):
    """1D convolution layer"""
    def __init__(self,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 trainable=True,
                 input_type="1d",
                 scope="conv1d"):
        """initialize 1d convolution layer"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.num_filter = num_filter
            self.window_size = window_size
            self.stride_size = stride_size
            self.padding_type = padding_type
            self.input_type = input_type
            self.trainable = trainable
            
            self.conv1d_layer = tf.layers.Conv1D(filters=self.num_filter,
                kernel_size=window_size, strides=stride_size, padding=self.padding_type, trainable=trainable)
    
    def __call__(self,
                 input_data):
        """generate 1d convolution layer output"""
        if self.input_type == "1d":
            input_conv1d = self.conv1d_layer(input_data)
        elif self.input_type == "2d":
            (batch_size, dim1_length, dim2_length,
                 input_embed_dim) = tf.shape(input_data)
            input_data = tf.reshape(input_data,
                shape=[batch_size * dim1_length, dim2_length, input_embed_dim])
            input_conv1d = self.conv1d_layer(input_data)
            _, dim2_length, input_embed_dim = tf.shape(input_conv1d)
            input_conv1d = tf.reshape(input_conv1d,
                shape=[batch_size, dim1_length, dim2_length, input_embed_dim])
        elif self.input_type == "3d":
            (batch_size, dim1_length, dim2_length, dim3_length,
                 input_embed_dim) = tf.shape(input_data)
            input_data = tf.reshape(input_data,
                shape=[batch_size * dim1_length * dim2_length, dim3_length, input_embed_dim])
            input_conv1d = self.conv1d_layer(input_data)
            _, dim3_length, input_embed_dim = tf.shape(input_conv1d)
            input_conv1d = tf.reshape(input_conv1d,
                shape=[batch_size, dim1_length, dim2_length, dim3_length, input_embed_dim])
        else:
            raise ValueError("unsupported input type {0}".format(input_type))
        
        return input_conv1d
