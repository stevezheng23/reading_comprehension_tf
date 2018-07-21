import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["RNN", "BiRNN"]

def _create_single_reccurent_cell(unit_dim,
                                  cell_type,
                                  activation,
                                  dropout,
                                  forget_bias,
                                  residual_connect,
                                  device_spec):
    """create single recurrent cell"""
    weight_initializer = create_variable_initializer("glorot_uniform")
    bias_initializer = create_variable_initializer("zero")
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
    
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=1.0-dropout)
    
    if residual_connect == True:
        single_cell = tf.contrib.rnn.ResidualWrapper(cell=single_cell)
    
    if device_spec is not None:
        single_cell = tf.contrib.rnn.DeviceWrapper(cell=single_cell, device=device_spec)
    
    return single_cell

def _creat_recurrent_cell(num_layer,
                          unit_dim,
                          cell_type,
                          activation,
                          dropout,
                          forget_bias,
                          residual_connect,
                          num_gpus,
                          default_gpu_id,
                          enable_multi_gpu):
    """create recurrent cell"""
    if num_layer > 1:
        cell_list = []
        for i in range(num_layer):
            if enable_multi_gpu == True:
                device_spec = get_device_spec(default_gpu_id + i, num_gpus)
            else:
                device_spec = get_device_spec(default_gpu_id, num_gpus)
            
            single_cell = _create_single_reccurent_cell(unit_dim, cell_type, activation,
                dropout, forget_bias, residual_connect, device_spec)
        cell_list.append(single_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    else:
        device_spec = get_device_spec(default_gpu_id, num_gpus)
        cell = _create_single_reccurent_cell(unit_dim, cell_type, activation,
            dropout, forget_bias, residual_connect, device_spec)
    
    return cell

class RNN(object):
    """uni-directional recurrent layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 cell_type,
                 activation,
                 dropout,
                 forget_bias=1.0,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="rnn"):
        """initialize uni-directional recurrent layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.cell_type = cell_type
        self.activation = activation
        self.dropout = dropout
        self.forget_bias = forget_bias
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.cell = _creat_recurrent_cell(self.num_layer, self.unit_dim, self.cell_type,
                self.activation, self.dropout, self.forget_bias, self.residual_connect,
                self.num_gpus, self.default_gpu_id, self.enable_multi_gpu)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call uni-directional recurrent layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_length = tf.cast(tf.reduce_sum(tf.squeeze(input_mask, axis=-1), axis=-1), dtype=tf.int32)
            output_recurrent, final_state_recurrent = tf.nn.dynamic_rnn(cell=self.cell,
                inputs=input_data, sequence_length=input_length, dtype=input_data.dtype)
            output_mask = input_mask
        
        return output_recurrent, output_mask

class BiRNN(object):
    """bi-directional recurrent layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 cell_type,
                 activation,
                 dropout,
                 forget_bias=1.0,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 enable_multi_gpu=True,
                 trainable=True,
                 scope="rnn"):
        """initialize bi-directional recurrent layer"""
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.cell_type = cell_type
        self.activation = activation
        self.dropout = dropout
        self.forget_bias = forget_bias
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.enable_multi_gpu = enable_multi_gpu
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            self.fwd_cell = _creat_recurrent_cell(self.num_layer, self.unit_dim, self.cell_type,
                self.activation, self.dropout, self.forget_bias, self.residual_connect,
                self.num_gpus, self.default_gpu_id, self.enable_multi_gpu)
            self.bwd_cell = _creat_recurrent_cell(self.num_layer, self.unit_dim, self.cell_type,
                self.activation, self.dropout, self.forget_bias, self.residual_connect,
                self.num_gpus, self.default_gpu_id + self.num_layer, self.enable_multi_gpu)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call bi-directional recurrent layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_length = tf.cast(tf.reduce_sum(tf.squeeze(input_mask, axis=-1), axis=-1), dtype=tf.int32)
            output_recurrent, final_state_recurrent = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fwd_cell,
                cell_bw=self.bwd_cell, inputs=input_data, sequence_length=input_length, dtype=input_data.dtype)
            
            output_recurrent = tf.concat(output_recurrent, -1)
            output_mask = input_mask
        
        return output_recurrent, output_mask

class GatedAttentionCellWrapper(RNNCell):
    def __init__(self,
                 cell,
                 attn_length,
                 attn_size=None,
                 attn_vec_size=None,
                 input_size=None,
                 state_is_tuple=True,
                 reuse=None):
    super(GatedAttentionCellWrapper, self).__init__(_reuse=reuse)
    
    if attn_length <= 0:
        raise ValueError("attention length should be greater than 0")
    
    if attn_size is None:
        attn_size = cell.output_size
    if attn_vec_size is None:
        attn_vec_size = attn_size
    
    self._cell = cell
    self._attn_length = attn_length
    self._attn_size = attn_size
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size
    self._state_is_tuple = state_is_tuple
    self._reuse = reuse
    
    @property
    def state_size(self):
        size = (self._cell.state_size, self._attn_size,
            self._attn_size * self._attn_length)
        
        if self._state_is_tuple:
            return size
        else:
            return sum(list(size))
    
    @property
    def output_size(self):
        return self._attn_size
    
    def call(self,
             inputs,
             state):
        pass
    
    def _attention(self,
                   query,
                   attn_states):
        pass
