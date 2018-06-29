import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["Dropout", "LayerNorm", "PositionalEncoding"]

class Dropout(object):
    """dropout layer"""
    def __init__(self,
                 keep_prob,
                 num_gpus=0,
                 default_gpu_id=0,
                 scope="dropout"):
        """initialize dropout layer"""
        self.keep_prob = keep_prob
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call dropout layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if self.keep_prob < 1.0:
                output_dropout = tf.nn.dropout(input_data, self.keep_prob)
            else:
                output_dropout = input_data
            
            output_mask = input_mask
            output_dropout = output_dropout * output_mask
        
        return output_dropout, output_mask

class LayerNorm(object):
    """layer norm layer"""
    def __init__(self,
                 layer_dim,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="layer_norm"):
        """initialize layer norm layer"""
        self.layer_dim = layer_dim
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            gamma_initializer = create_variable_initializer("glorot_uniform")
            beta_initializer = create_variable_initializer("zero")
            self.gamma = tf.get_variable("gamma", shape=[self.layer_dim],
                initializer=gamma_initializer, trainable=self.trainable, dtype=tf.float32)
            self.beta = tf.get_variable("beta", shape=[self.layer_dim],
                initializer=beta_initializer, trainable=self.trainable, dtype=tf.float32)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call layer norm layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data = input_data * input_mask
            input_mean, input_variance = tf.nn.moments(input_data, axes=[-1], keep_dims=True)
            output_norm = (input_data - input_mean) / tf.sqrt(input_variance + EPSILON)
            output_norm = output_norm * self.gamma + self.beta 
            output_mask = input_mask
            output_norm = output_norm * output_mask
        
        return output_norm, output_mask

class PositionalEncoding(object):
    """positional encoding layer"""
    def __init__(self,
                 max_length,
                 unit_dim,
                 time_scale,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="layer_norm"):
        """initialize positional encoding layer"""
        self.max_length = max_length
        self.unit_dim = unit_dim
        self.time_scale = time_scale
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call positional encoding layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_signal = input_data * input_mask
            input_signal_mask = input_mask
            
            num_time_scale = self.unit_dim / 2
            position = tf.to_float(tf.range(self.max_length))
            log_time_scale = np.log(float(self.time_scale)) / (float(num_time_scale) - 1)
            inv_time_scale = tf.exp(-1.0 * log_time_scale * tf.to_float(tf.range(num_time_scale)))
            scaled_time = tf.expand_dims(position, axis=1) * tf.expand_dims(inv_time_scale, axis=0)
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            signal = tf.pad(signal, paddings=[[0, 0], [0, self.unit_dim % 2]])
            signal = tf.reshape(signal, shape=[1, self.max_length, self.unit_dim])
            
            output_signal = input_signal + signal
            output_mask = input_signal_mask
            output_signal = output_signal * output_mask
        
        return output_signal, output_mask
