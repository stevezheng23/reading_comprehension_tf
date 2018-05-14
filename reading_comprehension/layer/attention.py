import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Attention"]

def _create_attention_matrix(src_unit_dim,
                             trg_unit_dim,
                             attention_unit_dim,
                             attention_score_type):
    """create attetnion matrix"""
    variable_name = "{0}_attention_matrix".format(attention_score_type)
    if attention_score_type == "multiplicative":
        attention_matrix = tf.get_variable(variable_name,
            shape=[src_unit_dim, trg_unit_dim], dtype=tf.float32)
    elif attention_score_type == "additive":
        attention_matrix = tf.get_variable(variable_name,
            shape=[attention_unit_dim, src_unit_dim + trg_unit_dim], dtype=tf.float32)
    else:
        raise ValueError("unsupported attention score type {0}".format(attention_score_type))
    
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
    input_trg_shape = tf.shape(input_trg_data)
    trg_batch_size = input_src_shape[0]
    trg_max_length = input_trg_shape[1]
    trg_unit_dim = input_trg_shape[2]
    input_trg_transpose = tf.transpose(input_trg_data,
        perm=[trg_batch_size, trg_unit_dim, trg_max_length])
    input_attention = tf.matmul(input_src_data, attention_matrix)
    input_attention = tf.matmul(input_attention, input_trg_transpose)
    
    return input_attention

def _generate_additive_attention_score(input_src_data,
                                       input_trg_data,
                                       attention_matrix):
    """generate additive attention score"""
    input_src_shape = tf.shape(input_src_data)
    input_trg_shape = tf.shape(input_trg_data)
    input_src_max_length = input_src_shape[1]
    input_trg_max_length = input_trg_shape[1]
    input_src_data = tf.expand_dims(input_src_data, axis=2)
    input_trg_data = tf.expand_dims(input_trg_data, axis=1)
    input_src_data = tf.tile(input_src_data, multiples=[1, 1, input_trg_max_length, 1])
    input_trg_data = tf.tile(input_trg_data, multiples=[1, input_src_max_length, 1, 1])
    input_attention = tf.concate([input_src_data, input_trg_data], axis=-1)
    input_attention = tf.matmul(input_attention, attention_matrix)
    
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
            input_attention_weight = tf.nn.softmax(input_attention_score, axis=-1)
            output_attention = tf.matmul(input_attention_weight, input_trg_data)
            
            return output_attention
