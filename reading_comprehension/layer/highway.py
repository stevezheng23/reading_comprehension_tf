import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

from layer.basic import *

__all__ = ["Highway", "ConvHighway", "StackedHighway", "StackedConvHighway"]

class Highway(object):
    """highway layer"""
    def __init__(self,
                 unit_dim,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="highway"):
        """initialize highway layer"""
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = create_variable_initializer("glorot_uniform", self.random_seed)
            bias_initializer = create_variable_initializer("zero")
            transform_activation = create_activation_function(self.activation)
            gate_activation = create_activation_function("sigmoid")
            self.transform_layer = tf.layers.Dense(units=self.unit_dim, activation=transform_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, trainable=self.trainable)
            self.gate_layer = tf.layers.Dense(units=self.unit_dim, activation=gate_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, trainable=self.trainable)
            
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            transform, _ = self.dropout_layer(self.transform_layer(input_data), input_mask)
            gate = self.gate_layer(input_data)
            output_highway = transform * gate + input_data * (1 - gate)
            output_mask = input_mask
        
        return output_highway, output_mask

class ConvHighway(object):
    """convolutional highway layer"""
    def __init__(self,
                 num_filter,
                 window_size,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="conv_highway"):
        """initialize convolutional highway layer"""
        self.num_filter = num_filter
        self.window_size = window_size
        self.activation = activation
        self.dropout = dropout
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = create_variable_initializer("glorot_uniform", self.m_seed)
            bias_initializer = create_variable_initializer("zero")
            transform_activation = create_activation_function(self.activation)
            gate_activation = create_activation_function("sigmoid")
            
            self.transform_layer = tf.layers.Conv1D(filters=self.num_filter, kernel_size=window_size,
                strides=1, padding="SAME", activation=transform_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, trainable=trainable)
            self.gate_layer = tf.layers.Conv1D(filters=self.num_filter, kernel_size=window_size,
                strides=1, padding="SAME", activation=gate_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, trainable=trainable)
            
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call convolutional highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            transform, _ = self.dropout_layer(self.transform_layer(input_data), input_mask)
            gate = self.gate_layer(input_data)
            output_highway = transform * gate + input_data * (1 - gate)
            output_mask = input_mask
        
        return output_highway, output_mask

class StackedHighway(object):
    """stacked highway layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_highway"):
        """initialize stacked highway layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.highway_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id
                sublayer_dropout = self.dropout[i] if self.dropout != None else 0.0
                highway_layer = Highway(unit_dim=self.unit_dim, activation=self.activation,
                    dropout=sublayer_dropout, num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id,
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.highway_layer_list.append(highway_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_highway = input_data
            input_highway_mask = input_mask
            
            for highway_layer in self.highway_layer_list:
                input_highway, input_highway_mask = highway_layer(input_highway, input_highway_mask)
            
            output_highway = input_highway
            output_mask = input_highway_mask
        
        return output_highway, output_mask

class StackedConvHighway(object):
    """stacked convolution highway layer"""
    def __init__(self,
                 num_layer,
                 num_filter,
                 window_size,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_conv_highway"):
        """initialize stacked convolution highway layer"""
        self.num_layer = num_layer
        self.num_filter = num_filter
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.highway_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id
                sublayer_dropout = self.dropout[i] if self.dropout != None else 0.0
                highway_layer = ConvHighway(num_filter=self.num_filter, window_size=self.window_size,
                    activation=self.activation, dropout=sublayer_dropout, num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, 
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.highway_layer_list.append(highway_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked convolution highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_highway = input_data
            input_highway_mask = input_mask
            
            for highway_layer in self.highway_layer_list:
                input_highway, input_highway_mask = highway_layer(input_highway, input_highway_mask)
            
            output_highway = input_highway
            output_mask = input_highway_mask
        
        return output_highway, output_mask
