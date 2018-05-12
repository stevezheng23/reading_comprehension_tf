import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["RNN", "BiRNN"]

def _create_single_reccurent_cell(unit_dim,
                                  cell_type,
                                  activation,
                                  drop_out,
                                  forget_bias,
                                  residual_connect,
                                  device_spec):
    """create single recurrent cell"""
    weight_initializer = create_variable_initializer("glorot_uniform")
    bias_initializer = create_variable_initializer("glorot_uniform")
    recurrent_activation = create_activation_function(activation)

    if cell_type == "lstm":
        single_cell = tf.contrib.rnn.LSTMCell(num_units=unit_dim, use_peepholes=False,
            activation=recurrent_activation, forget_bias=forget_bias, initializer=weight_initializer)
    elif cell_type == "peephole_lstm":
        single_cell = tf.contrib.rnn.LSTMCell(num_units=unit_dim, use_peepholes=True,
            activation=recurrent_activation, forget_bias=forget_bias, initializer=weight_initializer)
    elif cell_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=unit_dim, layer_norm=True,
            activation=recurrent_activation, forget_bias=forget_bias)
    elif cell_type == "block_lstm":
        single_cell = tf.contrib.rnn.LSTMBlockCell(num_units=unit_dim, forget_bias=forget_bias)
    elif cell_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units=unit_dim, activation=recurrent_activation,
            kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
    else:
        raise ValueError("unsupported cell type {0}".format(cell_type))
    
    if drop_out > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=1.0-drop_out)
    
    if residual_connect == True:
        single_cell = tf.contrib.rnn.ResidualWrapper(cell=single_cell)
    
    if device_spec is not None:
        single_cell = tf.contrib.rnn.DeviceWrapper(cell=single_cell, device=device_spec)
    
    return single_cell

def _creat_recurrent_cell(num_layer,
                          unit_dim,
                          cell_type,
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
            single_cell = _create_single_reccurent_cell(unit_dim, cell_type, activation,
                drop_out, forget_bias, residual_connect, get_device_spec(default_gpu_id+i, num_gpus))
        cell_list.append(single_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    else:
        cell = _create_single_reccurent_cell(unit_dim, cell_type, activation,
            drop_out, forget_bias, residual_connect, get_device_spec(default_gpu_id, num_gpus))
    
    return cell

class RNN(object):
    """uni-directional recurrent layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 cell_type,
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
        self.cell_type = cell_type
        self.activation = activation
        self.drop_out = drop_out
        self.forget_bias = forget_bias
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.cell = _creat_recurrent_cell(self.num_layer,
                self.unit_dim, self.cell_type, self.activation, self.drop_out,
                self.forget_bias, self.residual_connect, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call uni-directional recurrent layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_length = tf.cast(tf.reduce_sum(input_mask, axis=-1), dtype=tf.int32)
            output_recurrent, final_state_recurrent = tf.nn.dynamic_rnn(cell=self.cell,
                inputs=input_data, sequence_length=input_length, dtype=input_data.dtype)
            
            return output_recurrent, final_state_recurrent

class BiRNN(object):
    """bi-directional recurrent layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 cell_type,
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
        self.cell_type = cell_type
        self.activation = activation
        self.drop_out = drop_out
        self.forget_bias = forget_bias
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.fwd_cell = _creat_recurrent_cell(self.num_layer,
                self.unit_dim, self.cell_type, self.activation, self.drop_out,
                self.forget_bias, self.residual_connect, self.num_gpus, self.default_gpu_id)
            self.bwd_cell = _creat_recurrent_cell(self.num_layer,
                self.unit_dim, self.cell_type, self.activation, self.drop_out,
                self.forget_bias, self.residual_connect, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call bi-directional recurrent layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_length = tf.cast(tf.reduce_sum(input_mask, axis=-1), dtype=tf.int32)
            output_recurrent, final_state_recurrent = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fwd_cell,
                cell_bw=self.bwd_cell, inputs=input_data, sequence_length=input_length, dtype=input_data.dtype)
            
            output_recurrent = tf.concat(output_recurrent, -1)
            if self.num_layer > 1:
                state_list = []
                for i in range(self.num_layer):
                    state_list.append(final_state_recurrent[0][i])
                    state_list.append(final_state_recurrent[1][i])
                final_state_recurrent = tuple(state_list)
            
            return output_recurrent, final_state_recurrent
