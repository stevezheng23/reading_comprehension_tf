import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["create_variable_initializer", "create_weight_regularizer", "create_activation_function",
           "softmax_with_mask", "generate_masked_logit", "generate_onehot_label"]

def create_variable_initializer(initializer_type,
                                random_seed=None,
                                data_type=tf.float32):
    """create variable initializer"""
    if initializer_type == "zero":
        initializer = tf.zeros_initializer
    elif initializer_type == "orthogonal":
        initializer = tf.orthogonal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "random_uniform":
        initializer = tf.random_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_uniform":
        initializer = tf.glorot_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "xavier_uniform":
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=random_seed, dtype=tf.float32)
    elif initializer_type == "random_normal":
        initializer = tf.random_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "truncated_normal":
        initializer = tf.truncated_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_normal":
        initializer = tf.glorot_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "xavier_normal":
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=random_seed, dtype=tf.float32)
    else:
        initializer = None
    
    return initializer

def create_weight_regularizer(regularizer_type,
                              scale):
    """create weight regularizer"""
    if regularizer_type == "l1":
        regularizer = tf.contrib.layers.l1_regularizer(scale)
    elif regularizer_type == "l2":
        regularizer = tf.contrib.layers.l2_regularizer(scale)
    else:
        regularizer = None
    
    return regularizer

def create_activation_function(activation):
    """create activation function"""
    if activation == "tanh":
        activation_function = tf.nn.tanh
    elif activation == "relu":
        activation_function = tf.nn.relu
    elif activation == "leaky_relu":
        activation_function = tf.nn.leaky_relu
    elif activation == "sigmoid":
        activation_function = tf.nn.sigmoid
    else:
        activation_function = None
    
    return activation_function

def softmax_with_mask(input_data,
                      input_mask,
                      axis=-1,
                      keepdims=True):
    """compute softmax with masking"""    
    return tf.nn.softmax(input_data + MIN_FLOAT * (1 - input_mask), dim=axis)

def generate_masked_logit(input_data,
                          input_mask):
    """generate masked logit"""
    return input_data + MIN_FLOAT * (1 - input_mask)

def generate_onehot_label(input_data,
                          input_depth):
    """generate one-hot label"""
    return tf.one_hot(input_data, depth=input_depth, on_value=1.0, off_value=0.0, dtype=tf.float32)
