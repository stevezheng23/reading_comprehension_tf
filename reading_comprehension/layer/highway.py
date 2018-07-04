import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

from layer.basic import *

__all__ = ["Highway", "StackedHighway"]

class Highway(object):
    """highway layer"""
    def __init__(self,
                 unit_dim,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="highway"):
        """initialize highway layer"""
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("zero")
            transform_activation = create_activation_function(self.activation)
            gate_activation = create_activation_function("sigmoid")
            self.transform_layer = tf.layers.Dense(units=self.unit_dim, activation=transform_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=self.trainable)
            self.gate_layer = tf.layers.Dense(units=self.unit_dim, activation=gate_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=self.trainable)
            
            if self.dropout > 0.0:
                self.dropout_layer = Dropout(keep_prob=1.0-self.dropout,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if self.dropout > 0.0:
                input_data, input_mask = self.dropout_layer(input_data, input_mask)
            
            transform = self.transform_layer(input_data)
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
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="stacked_highway"):
        """initialize stacked highway layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.highway_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i if self.enable_multi_gpu == True else self.default_gpu_id
                highway_layer = Highway(unit_dim=self.unit_dim, activation=self.activation, dropout=self.dropout,
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, trainable=self.trainable, scope=layer_scope)
                self.highway_layer_list.append(highway_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_highway = input_data
            input_highway_mask = input_mask
            
            for highway_layer in self.highway_layer_list:
                input_highway, input_highway_mask = highway_layer(input_highway, input_highway_mask)
            
            output_highway = input_highway
            output_mask = input_highway_mask
        
        return output_highway, output_mask
