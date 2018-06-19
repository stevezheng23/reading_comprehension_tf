import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

from layer.basic import *

__all__ = ["Conv1D", "Conv2D", "MultiConv1D", "MultiConv2D"]

class Conv(object):
    """convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="conv"):
        """initialize convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.trainable = trainable
        self.scope=scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("zero")
            conv_activation = create_activation_function(self.activation)
            self.conv_layer = tf.layers.Conv1D(filters=self.num_filter, kernel_size=window_size,
                strides=stride_size, padding=self.padding_type, activation=conv_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=trainable)
            
            if self.dropout > 0.0:
                self.dropout_layer = Dropout(keep_prob=1.0-self.dropout,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id)
            
            if self.layer_norm == True:
                self.norm_layer = LayerNorm(layer_dim=self.num_channel,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=self.trainable)

class Conv1D(Conv):
    """1d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="conv1d"):
        """initialize 1d convolution layer"""
        super(Conv1D, self).__init__(num_channel=num_channel, num_filter=num_filter, window_size=window_size,
            stride_size=stride_size, padding_type=padding_type, activation=activation, dropout=dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data = input_data * input_mask
            input_conv = input_data
            input_conv_mask = input_mask
            
            if self.dropout > 0.0:
                input_conv, input_conv_mask = self.dropout_layer(input_conv, input_conv_mask)
            
            if self.layer_norm == True:
                input_conv, input_conv_mask = self.norm_layer(input_conv, input_conv_mask)
            
            input_conv = self.conv_layer(input_conv)
            
            if self.residual_connect == True:
                output_conv = input_conv + input_data
                output_mask = input_conv_mask * input_mask
            else:
                output_conv = input_conv
                output_mask = input_mask
            
            output_conv = output_conv * output_mask
        
        return output_conv, output_mask

class Conv2D(Conv):
    """2d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="conv2d"):
        """initialize 2d convolution layer"""
        super(Conv2D, self).__init__(num_channel=num_channel, num_filter=num_filter, window_size=window_size,
            stride_size=stride_size, padding_type=padding_type, activation=activation, dropout=dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 2d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data = input_data * input_mask
            input_conv = input_data
            input_conv_mask = input_mask
            
            if self.dropout > 0.0:
                input_conv, input_conv_mask = self.dropout_layer(input_conv, input_conv_mask)
            
            if self.layer_norm == True:
                input_conv, input_conv_mask = self.norm_layer(input_conv, input_conv_mask)
            
            input_conv_shape = tf.shape(input_conv)
            batch_size = input_conv_shape[0]
            max_length = input_conv_shape[-2]
            input_conv = tf.reshape(input_conv, shape=[-1, max_length, self.num_channel])
            input_conv = self.conv_layer(input_conv)
            input_conv_shape = tf.shape(input_conv)
            max_length = input_conv_shape[-2]
            input_conv = tf.reshape(input_conv, shape=[batch_size, -1, max_length, self.num_filter])
            
            if self.residual_connect == True:
                output_conv = input_conv + input_data
                output_mask = input_conv_mask * input_mask
            else:
                output_conv = input_conv
                output_mask = input_mask
            
            output_conv = output_conv * output_mask
        
        return output_conv, output_mask

class MultiConv1D(object):
    """multi-window 1d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="multi_conv1d"):
        """initialize multi-window 1d convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.conv_layer_list = []
            for i in range(len(self.window_size)):
                layer_scope = "layer_{0}".format(i)
                conv_layer = Conv1D(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size[i], stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id+i, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call multi-window 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_conv_list = []
            input_conv_mask_list = []
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_data, input_mask)
                input_conv_list.append(input_conv)
                input_conv_mask_list.append(input_conv_mask)
            
            output_conv = tf.concat(input_conv_list, axis=-1)
            output_mask = tf.reduce_max(tf.concat(input_conv_mask_list, axis=-1), axis=-1, keep_dims=True)
            output_conv = output_conv * output_mask
        
        return output_conv, output_mask

class MultiConv2D(object):
    """multi-window 2d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="multi_conv2d"):
        """initialize multi-window 2d convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.conv_layer_list = []
            for i in range(len(self.window_size)):
                layer_scope = "layer_{0}".format(i)
                conv_layer = Conv2D(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size[i], stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id+i, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call multi-window 2d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_conv_list = []
            input_conv_mask_list = []
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_data, input_mask)
                input_conv_list.append(input_conv)
                input_conv_mask_list.append(input_conv_mask)
            
            output_conv = tf.concat(input_conv_list, axis=-1)
            output_mask = tf.reduce_max(tf.concat(input_conv_mask_list, axis=-1), axis=-1, keep_dims=True)
            output_conv = output_conv * output_mask
        
        return output_conv, output_mask
