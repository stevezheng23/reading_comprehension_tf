import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["Dense", "StackedDense"]

class Dense(object):
    """dense layer"""
    def __init__(self,
                 unit_dim,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="dense"):
        """initialize dense layer"""
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("glorot_uniform")
            dense_activation = create_activation_function(self.activation)
            self.dense_layer = tf.layers.Dense(units=self.unit_dim, activation=dense_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=self.trainable)
    
    def __call__(self,
                 input_data):
        """call dense layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if self.dropout > 0.0:
                input_data = tf.nn.dropout(input_data, 1.0-self.dropout)
            
            output_dense = self.dense_layer(input_data)
        
        return output_dense

class StackedDense(object):
    """stacked dense layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="stacked_dense"):
        """initialize stacked dense layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.dense_layer_list = []
            for i in range(num_layer):
                layer_scope = "layer_{0}".format(i)
                dense_layer = Dense(unit_dim=self.unit_dim, activation=self.activation, dropout=self.dropout,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id+i, trainable=self.trainable, scope=layer_scope)
                self.dense_layer_list.append(dense_layer)
    
    def __call__(self,
                 input_data):
        """call stacked dense layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_dense = input_data
            for dense_layer in self.dense_layer_list:
                input_dense = dense_layer(input_dense)
            
            output_dense = input_dense
        
        return output_dense
