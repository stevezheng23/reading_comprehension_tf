import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Pooling1D"]

class Pooling1D(object):
    """pooling layer"""
    def __init__(self,
                 window_size,
                 stride_size,
                 padding_type,
                 pooling_type="max",
                 input_type="1d",
                 scope="conv1d"):
        """initialize 1d convolution layer"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.window_size = window_size
            self.stride_size = stride_size
            self.padding_type = padding_type
            self.input_type = input_type
            
            if pooling_type == "max":
                self.pooling1d_layer = tf.layers.MaxPooling1D(pool_size=self.window_size,
                    strides=stride_size, padding=self.padding_type)
            elif pooling_type == "avg":
                self.pooling1d_layer = tf.layers.AveragePooling1D(pool_size=self.window_size,
                    strides=stride_size, padding=self.padding_type)
            else:
                raise ValueError("unsupported pooling type {0}".format(pooling_type))

    
    def __call__(self,
                 input_data):
        """generate 1d pooling layer output"""
        if self.input_type == "1d":
            input_pooling1d = self.pooling1d_layer(input_data)
        elif self.input_type == "2d":
            (batch_size, dim1_length, dim2_length,
                 input_embed_dim) = tf.shape(input_data)
            input_data = tf.reshape(input_data,
                shape=[batch_size * dim1_length, dim2_length, input_embed_dim])
            input_pooling1d = self.pooling1d_layer(input_data)
            _, dim2_length, input_embed_dim = tf.shape(input_pooling1d)
            input_pooling1d = tf.reshape(input_pooling1d,
                shape=[batch_size, dim1_length, dim2_length, input_embed_dim])
        elif self.input_type == "3d":
            (batch_size, dim1_length, dim2_length, dim3_length,
                 input_embed_dim) = tf.shape(input_data)
            input_data = tf.reshape(input_data,
                shape=[batch_size * dim1_length * dim2_length, dim3_length, input_embed_dim])
            input_pooling1d = self.pooling1d_layer(input_data)
            _, dim3_length, input_embed_dim = tf.shape(input_pooling1d)
            input_pooling1d = tf.reshape(input_pooling1d,
                shape=[batch_size, dim1_length, dim2_length, dim3_length, input_embed_dim])
        else:
            raise ValueError("unsupported input type {0}".format(input_type))
        
        return input_pooling1d
