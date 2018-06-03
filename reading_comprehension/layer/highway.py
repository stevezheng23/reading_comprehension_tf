import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Highway", "StackedHighway"]

class Highway(object):
    """highway layer"""
    def __init__(self,
                 unit_dim,
                 activation,
                 dropout,
                 trainable=True,
                 scope="highway"):
        """initialize highway layer"""
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("glorot_uniform")
            transform_activation = create_activation_function(self.activation)
            gate_activation = create_activation_function("sigmoid")
            self.transform_layer = tf.layers.Dense(units=self.unit_dim, activation=transform_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=self.trainable)
            self.gate_layer = tf.layers.Dense(units=self.unit_dim, activation=gate_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=self.trainable)
    
    def __call__(self,
                 input_data):
        """call highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if self.dropout > 0.0:
                input_data = tf.nn.dropout(input_data, 1.0-self.dropout)
            
            transform = self.transform_layer(input_data)
            gate = self.gate_layer(input_data)
            output_highway = transform * gate + input_data * (1 - gate)
        
        return output_highway

class StackedHighway(object):
    """stacked highway layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 activation,
                 dropout,
                 trainable=True,
                 scope="stacked_highway"):
        """initialize stacked highway layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.highway_layer_list = []
            for i in range(num_layer):
                layer_scope = "layer_{0}".format(i)
                highway_layer = Highway(unit_dim=self.unit_dim, activation=self.activation,
                    dropout=self.dropout, trainable=self.trainable, scope=layer_scope)
                self.highway_layer_list.append(highway_layer)
    
    def __call__(self,
                 input_data):
        """call stacked highway layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_highway = input_data
            for highway_layer in self.highway_layer_list:
                input_highway = highway_layer(input_highway)
            
            output_highway = input_highway
        
        return output_highway
