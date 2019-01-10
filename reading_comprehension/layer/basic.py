import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["Dropout", "LayerNorm"]

class Dropout(object):
    """dropout layer"""
    def __init__(self,
                 rate,
                 num_gpus=0,
                 default_gpu_id=0,
                 random_seed=0,
                 scope="dropout"):
        """initialize dropout layer"""
        self.rate = rate
        self.random_seed = random_seed
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call dropout layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if self.rate > 0.0:
                output_dropout = tf.layers.dropout(input_data, self.rate, seed=self.random_seed)
            else:
                output_dropout = input_data
            
            output_mask = input_mask
        
        return output_dropout, output_mask

class LayerNorm(object):
    """layer norm layer"""
    def __init__(self,
                 layer_dim,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="layer_norm"):
        """initialize layer norm layer"""
        self.layer_dim = layer_dim
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            gamma_initializer = create_variable_initializer("one")
            beta_initializer = create_variable_initializer("zero")
            self.gamma = tf.get_variable("gamma", shape=[self.layer_dim], initializer=gamma_initializer,
                regularizer=self.regularizer, trainable=self.trainable, dtype=tf.float32)
            self.beta = tf.get_variable("beta", shape=[self.layer_dim], initializer=beta_initializer,
                regularizer=self.regularizer, trainable=self.trainable, dtype=tf.float32)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call layer norm layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_mean, input_variance = tf.nn.moments(input_data, axes=[-1], keep_dims=True)
            output_norm = (input_data - input_mean) / tf.sqrt(input_variance + EPSILON)
            output_norm = output_norm * self.gamma + self.beta 
            output_mask = input_mask
        
        return output_norm, output_mask
