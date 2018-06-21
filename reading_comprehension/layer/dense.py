import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

from layer.basic import *

__all__ = ["Dense", "StackedDense"]

class Dense(object):
    """dense layer"""
    def __init__(self,
                 unit_dim,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="dense"):
        """initialize dense layer"""
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = create_variable_initializer("glorot_uniform")
            bias_initializer = create_variable_initializer("zero")
            self.dense_layer = tf.layers.Dense(units=self.unit_dim, activation=None, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer, trainable=self.trainable)
            
            self.dense_activation = create_activation_function(self.activation)
            
            if self.dropout > 0.0:
                self.dropout_layer = Dropout(keep_prob=1.0-self.dropout,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id)
            
            if self.layer_norm == True:
                self.norm_layer = LayerNorm(layer_dim=self.unit_dim,
                    num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=self.trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call dense layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data = input_data * input_mask
            input_dense = input_data
            input_dense_mask = input_mask
            
            if self.dropout > 0.0:
                input_dense, input_dense_mask = self.dropout_layer(input_dense, input_dense_mask)
            
            input_dense = self.dense_layer(input_dense)
            
            if self.layer_norm == True:
                input_dense, input_dense_mask = self.norm_layer(input_dense, input_dense_mask)
            
            if self.dense_activation != None:
                input_dense = self.dense_activation(input_dense)
            
            if self.residual_connect == True:
                output_dense = input_dense + input_data
                output_mask = input_dense_mask * input_mask
            else:
                output_dense = input_dense
                output_mask = input_dense_mask
            
            output_dense = output_dense * output_mask
        
        return output_dense, output_mask

class StackedDense(object):
    """stacked dense layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 activation,
                 dropout,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="stacked_dense"):
        """initialize stacked dense layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.dense_layer_list = []
            for i in range(num_layer):
                layer_scope = "layer_{0}".format(i)
                dense_layer = Dense(unit_dim=self.unit_dim, activation=self.activation, dropout=self.dropout,
                    layer_norm=self.layer_norm, residual_connect=self.residual_connect, num_gpus=self.num_gpus,
                    default_gpu_id=self.default_gpu_id+i, trainable=self.trainable, scope=layer_scope)
                self.dense_layer_list.append(dense_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked dense layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_dense = input_data
            for dense_layer in self.dense_layer_list:
                input_dense, input_mask = dense_layer(input_dense, input_mask)
            
            output_dense = input_dense
            output_mask = input_mask
        
        return output_dense, output_mask
