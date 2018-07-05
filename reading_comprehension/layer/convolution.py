import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

from layer.basic import *

__all__ = ["Conv1D", "Conv2D", "MultiConv1D", "MultiConv2D",
           "SeparableConv1D", "SeparableConv2D", "MultiSeparableConv1D", "MultiSeparableConv2D",
           "StackedConv", "StackedMultiConv", "StackedSeparableConv", "StackedMultiSeparableConv"]

class Conv1D(object):
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
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
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
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
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
        
        return output_conv, output_mask

class Conv2D(object):
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
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("zero")
            conv_activation = create_activation_function(self.activation)
            self.conv_layer = tf.layers.Conv2D(filters=self.num_filter, kernel_size=[1, window_size],
                strides=[1, stride_size], padding=self.padding_type, activation=conv_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=trainable)
            
            if self.dropout > 0.0:
                self.dropout_layer = Dropout(keep_prob=1.0-self.dropout,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id)
            
            if self.layer_norm == True:
                self.norm_layer = LayerNorm(layer_dim=self.num_channel,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=self.trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 2d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
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
                 enable_multi_gpu=True,
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
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(len(self.window_size)):
                layer_scope = "window_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = Conv1D(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size[i], stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call multi-window 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv_list = []
            input_conv_mask_list = []
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_data, input_mask)
                input_conv_list.append(input_conv)
                input_conv_mask_list.append(input_conv_mask)
            
            output_conv = tf.concat(input_conv_list, axis=-1)
            output_mask = tf.reduce_max(tf.concat(input_conv_mask_list, axis=-1), axis=-1, keep_dims=True)
        
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
                 enable_multi_gpu=True,
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
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(len(self.window_size)):
                layer_scope = "window_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = Conv2D(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size[i], stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call multi-window 2d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv_list = []
            input_conv_mask_list = []
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_data, input_mask)
                input_conv_list.append(input_conv)
                input_conv_mask_list.append(input_conv_mask)
            
            output_conv = tf.concat(input_conv_list, axis=-1)
            output_mask = tf.reduce_max(tf.concat(input_conv_mask_list, axis=-1), axis=-1, keep_dims=True)
        
        return output_conv, output_mask

class SeparableConv(object):
    """depthwise-separable convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 num_multiplier,
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
                 scope="sep_conv"):
        """initialize depthwise-separable convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.num_multiplier = num_multiplier
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
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("zero")
            self.depthwise_filter = tf.get_variable("depthwise_filter",
                shape=[1, self.window_size, self.num_channel, self.num_multiplier],
                initializer=weight_initializer, trainable=self.trainable, dtype=tf.float32)
            self.pointwise_filter = tf.get_variable("pointwise_filter",
                shape=[1, 1, self.num_channel * self.num_multiplier, self.num_filter],
                initializer=weight_initializer, trainable=self.trainable, dtype=tf.float32)
            self.separable_bias = tf.get_variable("separable_bias", shape=[self.num_filter],
                initializer=bias_initializer, trainable=trainable, dtype=tf.float32)
            
            self.strides = [1, 1, self.stride_size, 1]
            self.conv_activation = create_activation_function(self.activation)
            
            if self.dropout > 0.0:
                self.dropout_layer = Dropout(keep_prob=1.0-self.dropout,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id)
            
            if self.layer_norm == True:
                self.norm_layer = LayerNorm(layer_dim=self.num_channel,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=self.trainable)

class SeparableConv1D(SeparableConv):
    """depthwise-separable 1d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 num_multiplier,
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
                 scope="sep_conv1d"):
        """initialize depthwise-separable 1d convolution layer"""
        super(SeparableConv1D, self).__init__(num_channel=num_channel, num_filter=num_filter,
            num_multiplier=num_multiplier, window_size=window_size, stride_size=stride_size, padding_type=padding_type, 
            activation=activation, dropout=dropout, layer_norm=layer_norm, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call depthwise-separable 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            if self.dropout > 0.0:
                input_conv, input_conv_mask = self.dropout_layer(input_conv, input_conv_mask)
            
            if self.layer_norm == True:
                input_conv, input_conv_mask = self.norm_layer(input_conv, input_conv_mask)
            
            input_conv = tf.expand_dims(input_conv, axis=1)
            input_conv = tf.nn.separable_conv2d(input_conv, self.depthwise_filter,
                self.pointwise_filter, self.strides, self.padding_type)
            input_conv = tf.squeeze(input_conv, axis=1)
            
            input_conv = input_conv + self.separable_bias
            if self.conv_activation != None:
                input_conv = self.conv_activation(input_conv)
            
            if self.residual_connect == True:
                output_conv = input_conv + input_data
                output_mask = input_conv_mask * input_mask
            else:
                output_conv = input_conv
                output_mask = input_mask
        
        return output_conv, output_mask

class SeparableConv2D(SeparableConv):
    """depthwise-separable 2d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 num_multiplier,
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
                 scope="sep_conv2d"):
        """initialize depthwise-separable 2d convolution layer"""
        super(SeparableConv2D, self).__init__(num_channel=num_channel, num_filter=num_filter,
            num_multiplier=num_multiplier, window_size=window_size, stride_size=stride_size, padding_type=padding_type, 
            activation=activation, dropout=dropout, layer_norm=layer_norm, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call depthwise-separable 2d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            if self.dropout > 0.0:
                input_conv, input_conv_mask = self.dropout_layer(input_conv, input_conv_mask)
            
            if self.layer_norm == True:
                input_conv, input_conv_mask = self.norm_layer(input_conv, input_conv_mask)
            
            input_conv = tf.nn.separable_conv2d(input_conv, self.depthwise_filter,
                self.pointwise_filter, self.strides, self.padding_type)
            
            input_conv = input_conv + self.separable_bias
            input_conv = self.conv_activation(input_conv)
            
            if self.residual_connect == True:
                output_conv = input_conv + input_data
                output_mask = input_conv_mask * input_mask
            else:
                output_conv = input_conv
                output_mask = input_mask
        
        return output_conv, output_mask

class MultiSeparableConv1D(object):
    """multi-window depthwise-separable 1d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 num_multiplier,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="multi_sep_conv1d"):
        """initialize multi-window depthwise-separable 1d convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.num_multiplier = num_multiplier
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(len(self.window_size)):
                layer_scope = "window_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = SeparableConv1D(num_channel=self.num_channel, num_filter=self.num_filter, num_multiplier=self.num_multiplier,
                    window_size=self.window_size[i], stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call multi-window depthwise-separable 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv_list = []
            input_conv_mask_list = []
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_data, input_mask)
                input_conv_list.append(input_conv)
                input_conv_mask_list.append(input_conv_mask)
            
            output_conv = tf.concat(input_conv_list, axis=-1)
            output_mask = tf.reduce_max(tf.concat(input_conv_mask_list, axis=-1), axis=-1, keep_dims=True)
        
        return output_conv, output_mask

class MultiSeparableConv2D(object):
    """multi-window depthwise-separable 2d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 num_multiplier,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="multi_sep_conv2d"):
        """initialize multi-window depthwise-separable 2d convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.num_multiplier = num_multiplier
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(len(self.window_size)):
                layer_scope = "window_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = SeparableConv2D(num_channel=self.num_channel, num_filter=self.num_filter, num_multiplier=self.num_multiplier,
                    window_size=self.window_size[i], stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call multi-window depthwise-separable 2d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv_list = []
            input_conv_mask_list = []
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_data, input_mask)
                input_conv_list.append(input_conv)
                input_conv_mask_list.append(input_conv_mask)
            
            output_conv = tf.concat(input_conv_list, axis=-1)
            output_mask = tf.reduce_max(tf.concat(input_conv_mask_list, axis=-1), axis=-1, keep_dims=True)
        
        return output_conv, output_mask

class StackedConv(object):
    """stacked convolution layer"""
    def __init__(self,
                 layer_creator,
                 num_layer,
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
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="stacked_conv"):
        """initialize stacked convolution layer"""
        self.layer_creator = layer_creator
        self.num_layer = num_layer
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
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = self.layer_creator(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size, stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_conv, input_conv_mask)
            
            output_conv = input_conv
            output_mask = input_conv_mask
        
        return output_conv, output_mask

class StackedMultiConv(object):
    """stacked multi-window convolution layer"""
    def __init__(self,
                 layer_creator,
                 num_layer,
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
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="stacked_multi_conv"):
        """initialize stacked multi-window convolution layer"""
        self.layer_creator = layer_creator
        self.num_layer = num_layer
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
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = self.layer_creator(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size, stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm,
                    residual_connect=self.residual_connect, num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id,
                    enable_multi_gpu=self.enable_multi_gpu, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked multi-window convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_conv, input_conv_mask)
            
            output_conv = input_conv
            output_mask = input_conv_mask
        
        return output_conv, output_mask

class StackedSeparableConv(object):
    """stacked depthwise-separable convolution layer"""
    def __init__(self,
                 layer_creator,
                 num_layer,
                 num_channel,
                 num_filter,
                 num_multiplier,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="stacked_sep_conv"):
        """initialize stacked depthwise-separable convolution layer"""
        self.layer_creator = layer_creator
        self.num_layer = num_layer
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.num_multiplier = num_multiplier
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = self.layer_creator(num_channel=self.num_channel, num_filter=self.num_filter,
                    num_multiplier=self.num_multiplier, window_size=self.window_size, stride_size=self.stride_size,
                    padding_type=self.padding_type, activation=self.activation, dropout=self.dropout,
                    layer_norm=self.layer_norm, residual_connect=self.residual_connect, num_gpus=self.num_gpus,
                    default_gpu_id=layer_default_gpu_id, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked depthwise-separable convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_conv, input_conv_mask)
            
            output_conv = input_conv
            output_mask = input_conv_mask
        
        return output_conv, output_mask

class StackedMultiSeparableConv(object):
    """stacked depthwise-separable convolution layer"""
    def __init__(self,
                 layer_creator,
                 num_layer,
                 num_channel,
                 num_filter,
                 num_multiplier,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="stacked_sep_conv"):
        """initialize stacked depthwise-separable convolution layer"""
        self.layer_creator = layer_creator
        self.num_layer = num_layer
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.num_multiplier = num_multiplier
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.conv_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                conv_layer = self.layer_creator(num_channel=self.num_channel, num_filter=self.num_filter,
                    num_multiplier=self.num_multiplier, window_size=self.window_size, stride_size=self.stride_size,
                    padding_type=self.padding_type, activation=self.activation, dropout=self.dropout, layer_norm=self.layer_norm,
                    residual_connect=self.residual_connect, num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id,
                    enable_multi_gpu=self.enable_multi_gpu, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked depthwise-separable convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_conv, input_conv_mask)
            
            output_conv = input_conv
            output_mask = input_conv_mask
        
        return output_conv, output_mask
