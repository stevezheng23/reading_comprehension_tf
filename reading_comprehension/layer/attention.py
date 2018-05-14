import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Attention"]

def _create_attention_matrix(src_unit_dim,
                             trg_unit_dim,
                             attention_unit_dim,
                             attention_score_type):
    """create attetnion matrix"""
    if attention_score_type == "multiplicative":
        attention_matrix = _create_multiplicative_attention_matrix(src_unit_dim, trg_unit_dim)
    elif attention_score_type == "additive":
        attention_matrix = _create_additive_attention_matrix(src_unit_dim, trg_unit_dim, attention_unit_dim)
    else:
        raise ValueError("unsupported attention score type {0}".format(attention_score_type))
    
    return attention_matrix

def _create_multiplicative_attention_matrix(src_unit_dim,
                                            trg_unit_dim):
    """create multiplicative attetnion matrix"""
    attention_matrix = tf.get_variable("mul_att_mat",
        shape=[src_unit_dim, trg_unit_dim], dtype=tf.float32)
    attention_matrix = [attention_matrix]
    
    return attention_matrix

def _create_additive_attention_matrix(src_unit_dim,
                                      trg_unit_dim,
                                      attention_unit_dim):
    """create additive attetnion matrix"""
    pre_activation_matrix = tf.get_variable("add_att_mat_pre",
        shape=[attention_unit_dim, src_unit_dim + trg_unit_dim], dtype=tf.float32)
    post_activation_matrix = tf.get_variable("add_att_mat_post",
        shape=[1, attention_unit_dim], dtype=tf.float32)
    attention_matrix = [pre_activation_matrix, post_activation_matrix]
    
    return attention_matrix

def _generate_attention_score(input_src_data,
                              input_trg_data,
                              attention_matrix,
                              attention_score_type):
    """generate attention score"""
    if attention_score_type == "multiplicative":
        input_attention_score = _generate_multiplicative_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    elif attention_score_type == "additive":
        input_attention_score = _generate_additive_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    else:
        raise ValueError("unsupported attention score type {0}".format(attention_score_type))
    
    return input_attention_score

def _generate_multiplicative_attention_score(input_src_data,
                                             input_trg_data,
                                             attention_matrix):
    """generate multiplicative attention score"""
    input_src_shape = tf.shape(input_src_data)
    input_trg_shape = tf.shape(input_trg_data)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    src_dim = input_src_shape[-1]
    pre_activation_matrix = attention_matrix[0]
    input_trg_data = tf.transpose(input_trg_data, perm=[0, 2, 1])
    input_attention = tf.reshape(input_src_data, shape=[-1, src_dim])
    input_attention = tf.matmul(input_attention, pre_activation_matrix)
    input_attention = tf.reshape(input_src_data, shape=[batch_size, src_max_length, -1])
    input_attention = tf.matmul(input_attention, input_trg_data)
    
    return input_attention

def _generate_additive_attention_score(input_src_data,
                                       input_trg_data,
                                       attention_matrix):
    """generate additive attention score"""
    input_src_shape = tf.shape(input_src_data)
    input_trg_shape = tf.shape(input_trg_data)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    input_src_data = tf.expand_dims(input_src_data, axis=2)
    input_trg_data = tf.expand_dims(input_trg_data, axis=1)
    input_src_data = tf.tile(input_src_data, multiples=[1, 1, trg_max_length, 1])
    input_trg_data = tf.tile(input_trg_data, multiples=[1, src_max_length, 1, 1])
    pre_activation_matrix = tf.transpose(attention_matrix[0], perm=[1, 0])
    post_activation_matrix = tf.transpose(attention_matrix[1], perm=[1, 0])
    input_concat = tf.concat([input_src_data, input_trg_data], axis=-1)
    concat_dim = tf.shape(input_concat)[-1]
    input_attention = tf.reshape(input_concat, shape=[-1, concat_dim])
    input_attention = tf.matmul(input_attention, pre_activation_matrix)
    input_attention = tf.nn.tanh(input_attention)
    input_attention = tf.matmul(input_attention, post_activation_matrix)
    input_attention = tf.reshape(input_attention, shape=[batch_size, src_max_length, trg_max_length])
    
    return input_attention

class Attention(object):
    """attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 unit_dim,
                 score_type,
                 trainable=True,
                 scope="attention"):
        """initialize attention layer"""
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.unit_dim = unit_dim
        self.score_type = score_type
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.attention_matrix = _create_attention_matrix(self.src_dim,
                    self.trg_dim, self.unit_dim, self.score_type)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_attention_score = _generate_attention_score(input_src_data,
                input_trg_data, self.attention_matrix, self.score_type)
            input_attention_weight = tf.nn.softmax(input_attention_score, dim=-1)
            output_attention = tf.matmul(input_attention_weight, input_trg_data)
            
            return output_attention
