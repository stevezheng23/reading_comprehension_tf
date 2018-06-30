import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["SinusoidPosition"]

class SinusoidPosition(object):
    """sinusoid position layer"""
    def __init__(self,
                 unit_dim,
                 max_length,
                 time_scale,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="sin_pos"):
        """initialize sinusoid position layer"""
        self.unit_dim = unit_dim
        self.max_length = max_length
        self.time_scale = time_scale
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call sinusoid position layer"""
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
