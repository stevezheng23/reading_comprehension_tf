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
                 memory,
                 memory_mask,
                 attention_mechanism,
                 reuse=None,
                 regularizer=None,
                 trainable=True,
                 scope="gated_attention"):
        """initialize gated-attention cell wrapper"""
        super(GatedAttentionCellWrapper, self).__init__(_reuse=reuse)
        
        self._cell = cell
        self._memory = memory
        self._memory_mask = memory_mask
        self._attention_mechanism = attention_mechanism
        self._regularizer = regularizer
        self._trainable = trainable
        self._scope = scope
        
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE), tf.device('/CPU:0'):
            weight_initializer = create_variable_initializer("glorot_uniform")
            gate_activation = create_activation_function("sigmoid")
            self.gate_layer = tf.layers.Dense(units=self.unit_dim, activation=gate_activation,
                kernel_initializer=weight_initializer, kernel_regularizer=self._regularizer, trainable=self._trainable)
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def __call__(self,
                 inputs,
                 state):
        """call gated-attention cell wrapper"""
        inputs = self._attention(inputs, state)
        cell_output, new_state = self._cell(inputs, state)
        
        return cell_output, new_state
    
    def _attention(self,
                   inputs,
                   state):
        query = tf.concat([inputs, state], axis=-1)
        query_mask = tf.cast(tf.reduce_any(query, axis=-1), dtype=tf.float32)
        context, context_mask = self._attention_mechanism(query, query_mask, self._memory, self._memory_mask)
        attention = tf.concat([inputs, context], axis=-1)
        gate = self.gate_layer(attention)
        attention = gate * attention
        
        return attention
