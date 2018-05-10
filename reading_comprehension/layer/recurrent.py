import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["RNN", "BiRNN"]

def _create_single_reccurent_cell(unit_dim,
                                  unit_type,
                                  activation,
                                  drop_out,
                                  forget_bias,
                                  residual_connect,
                                  device_spec):
    """create single recurrent cell"""
    weight_initializer = create_variable_initializer("glorot_uniform")
    bias_initializer = create_variable_initializer("glorot_uniform")
    recurrent_activation = create_activation_function(activation)

    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.LSTMCell(num_units=unit_dim, use_peepholes=False,
            activation=recurrent_activation, forget_bias=forget_bias, initializer=weight_initializer)
    elif unit_type == "peephole_lstm":
        single_cell = tf.contrib.rnn.LSTMCell(num_units=unit_dim, use_peepholes=True,
            activation=recurrent_activation, forget_bias=forget_bias, initializer=weight_initializer)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=unit_dim, layer_norm=True,
            activation=recurrent_activation, forget_bias=forget_bias)
    elif unit_type == "block_lstm":
        single_cell = tf.contrib.rnn.LSTMBlockCell(num_units=unit_dim, forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units=unit_dim, activation=recurrent_activation,
            kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
    else:
        raise ValueError("unsupported unit type {0}".format(unit_type))
    
    if drop_out > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=1.0-drop_out)
    
    if residual_connect == True:
        single_cell = tf.contrib.rnn.ResidualWrapper(cell=single_cell)
    
    if device_spec is not None:
        single_cell = tf.contrib.rnn.DeviceWrapper(cell=single_cell, device=device_spec)
    
    return single_cell

def _creat_recurrent_cell(num_layer,
                          unit_dim,
                          unit_type,
                          activation,
                          drop_out,
                          forget_bias,
                          residual_connect,
                          num_gpus,
                          default_gpu_id):
    """create recurrent cell"""
    if num_layer > 1:
        cell_list = []
        for i in range(num_layer):
            single_cell = self._create_single_reccurent_cell(unit_dim, unit_type, activation,
                drop_out, forget_bias, residual_connect, get_device_spec(default_gpu_id+i, num_gpus))
        cell_list.append(single_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    else:
        cell = self._create_single_reccurent_cell(unit_dim, unit_type, activation,
            drop_out, forget_bias, residual_connect, get_device_spec(default_gpu_id, num_gpus))
    
    return cell

class RNN(object):
    """uni-directional recurrent layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 unit_type,
                 activation,
                 drop_out,
                 forget_bias=1.0,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="rnn"):
        """initialize uni-directional recurrent layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.unit_type = unit_type
        self.activation = activation
        self.drop_out = drop_out
        self.forget_bias = forget_bias
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.cell = _creat_recurrent_cell(self.num_layer,
                self.unit_dim, self.unit_type, self.activation, self.drop_out,
                self.forget_bias, self.residual_connect, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """generate uni-directional recurrent layer output"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_length = tf.reduce_sum(input_mask, axis=-1)
            output_recurrent, final_state_recurrent = tf.nn.dynamic_rnn(cell=self.cell,
                inputs=input_data, sequence_length=input_length)
            
            return output_recurrent, final_state_recurrent

class BiRNN(object):
    """bi-directional recurrent layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 unit_type,
                 activation,
                 drop_out,
                 forget_bias=1.0,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="rnn"):
        """initialize uni-directional recurrent layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.unit_type = unit_type
        self.activation = activation
        self.drop_out = drop_out
        self.forget_bias = forget_bias
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.fwd_cell = _creat_recurrent_cell(self.num_layer,
                self.unit_dim, self.unit_type, self.activation, self.drop_out,
                self.forget_bias, self.residual_connect, self.num_gpus, self.default_gpu_id)
            self.bwd_cell = _creat_recurrent_cell(self.num_layer,
                self.unit_dim, self.unit_type, self.activation, self.drop_out,
                self.forget_bias, self.residual_connect, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """generate bi-directional recurrent layer output"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_length = tf.reduce_sum(input_mask, axis=-1)
            output_recurrent, final_state_recurrent = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fwd_cell,
                cell_bw=bwd_cell, inputs=input_data, sequence_length=input_length)
            
            return output_recurrent, final_state_recurrent
