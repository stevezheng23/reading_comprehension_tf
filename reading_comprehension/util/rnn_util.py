import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["create_rnn_cell", "create_rnn_single_cell"]

def create_rnn_single_cell(unit_dim,
                           unit_type,
                           activation,
                           forget_bias,
                           residual_connect,
                           drop_out,
                           device_spec):
    """create single rnn cell"""
    activation_function = create_activation_function(activation)
    
    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.LSTMCell(num_units=unit_dim,
            activation=activation_function, forget_bias=forget_bias)
    elif unit_type == "peephole_lstm":
        single_cell = tf.contrib.rnn.LSTMCell(num_units=unit_dim,
            use_peepholes=True, activation=activation_function, forget_bias=forget_bias)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=unit_dim,
            layer_norm=True, activation=activation_function, forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units=unit_dim, activation=activation_function)
    else:
        raise ValueError("unsupported unit type {0}".format(unit_type))
    
    if drop_out > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=1.0-drop_out)
        
    if residual_connect == True:
        single_cell = tf.contrib.rnn.ResidualWrapper(cell=single_cell)
    
    if device_spec is not None:
        single_cell = tf.contrib.rnn.DeviceWrapper(cell=single_cell, device=device_spec)
    
    return single_cell

def create_rnn_cell(num_layer,
                    unit_dim,
                    unit_type,
                    activation,
                    forget_bias,
                    residual_connect,
                    drop_out,
                    num_gpus,
                    default_gpu_id):
    """create rnn cell"""
    if num_layer > 1:
        cell_list = []
        for i in range(num_layer):
            single_cell = create_rnn_single_cell(unit_dim=unit_dim, unit_type=unit_type, activation=activation,
                forget_bias=forget_bias, residual_connect=residual_connect, drop_out=drop_out,
                device_spec=get_device_spec(default_gpu_id+i, num_gpus))
            cell_list.append(single_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    else:
        cell = create_rnn_single_cell(unit_dim=unit_dim, unit_type=unit_type, activation=activation,
            forget_bias=forget_bias, residual_connect=residual_connect, drop_out=drop_out,
            device_spec=get_device_spec(default_gpu_id, num_gpus))
    
    return cell
