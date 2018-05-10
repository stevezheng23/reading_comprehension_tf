import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Conv1D", "Conv2D"]

class Conv(object):
    """convolution layer"""
    def __init__(self,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 trainable=True,
                 scope="conv"):
        """initialize convolution layer"""
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.trainable = trainable
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("glorot_uniform")
            conv_activation = create_activation_function(self.activation)
            self.conv_layer = tf.layers.Conv1D(filters=self.num_filter, kernel_size=window_size,
                strides=stride_size, padding=self.padding_type, activation=conv_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=trainable)

class Conv1D(Conv):
    """1d convolution layer"""
    def __init__(self,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 trainable=True,
                 scope="conv1d"):
        """initialize 1d convolution layer"""
        self.scope = scope
        
        super(Conv1D, self).__init__(num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, trainable=trainable, scope=self.scope)
    
    def __call__(self,
                 input_data):
        """generate 1d convolution layer output"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            output_conv = self.conv_layer(input_data)
            
            return output_conv

class Conv2D(Conv):
    """2d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 trainable=True,
                 scope="conv2d"):
        """initialize 2d convolution layer"""
        self.num_channel = num_channel
        self.scope = scope
        
        super(Conv2D, self).__init__(num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, trainable=trainable, scope=self.scope)
    
    def __call__(self,
                 input_data):
        """generate 2d convolution layer output"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_data_shape = tf.shape(input_data)
            batch_size = input_data_shape[0]
            dim1_length = input_data_shape[1]
            dim2_length = input_data_shape[2]
            input_data = tf.reshape(input_data,
                shape=[batch_size * dim1_length, dim2_length, self.num_channel])
            input_conv = self.conv_layer(input_data)
            input_conv_shape = tf.shape(input_conv)
            dim2_length = input_conv_shape[1]
            output_conv = tf.reshape(input_conv,
                shape=[batch_size, dim1_length, dim2_length, self.num_filter])
            
            return output_conv
