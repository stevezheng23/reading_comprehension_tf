import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

__all__ = ["Attention", "MaxAttention", "SelfAttention"]

def _create_attention_matrix(src_unit_dim,
                             trg_unit_dim,
                             attention_unit_dim,
                             attention_score_type):
    """create attetnion matrix"""
    if attention_score_type == "mul":
        attention_matrix = _create_multiplicative_attention_matrix(src_unit_dim, trg_unit_dim)
    elif attention_score_type == "add":
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
    if attention_score_type == "mul":
        input_attention_score = _generate_multiplicative_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    elif attention_score_type == "add":
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

def _generate_attention_mask(input_src_mask,
                             input_trg_mask,
                             remove_diag=False):
    """generate attention mask"""
    input_src_shape = tf.shape(input_src_mask)
    input_trg_shape = tf.shape(input_trg_mask)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    input_src_mask = tf.expand_dims(input_src_mask, axis=2)
    input_trg_mask = tf.expand_dims(input_trg_mask, axis=1)
    input_src_mask = tf.tile(input_src_mask, multiples=[1, 1, trg_max_length, 1])
    input_trg_mask = tf.tile(input_trg_mask, multiples=[1, src_max_length, 1, 1])
    input_mask = input_src_mask * input_trg_mask
    input_mask = tf.squeeze(input_mask, axis=-1)
    if remove_diag == True:
        input_mask = input_mask * (1 - tf.eye(src_max_length, trg_max_length))
    
    return input_mask

class Attention(object):
    """attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 unit_dim,
                 score_type,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="attention"):
        """initialize attention layer"""
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.unit_dim = unit_dim
        self.score_type = score_type
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.attention_matrix = _create_attention_matrix(self.src_dim,
                self.trg_dim, self.unit_dim, self.score_type)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_data = input_src_data * input_src_mask
            input_trg_data = input_trg_data * input_trg_mask
            input_attention_score = _generate_attention_score(input_src_data,
                input_trg_data, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_mask, input_trg_mask, False)
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1, keepdims=True)
            output_attention = tf.matmul(input_attention_weight, input_trg_data)
            output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def get_attention_matrix():
        return self.attention_matrix

class MaxAttention(object):
    """max-attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 unit_dim,
                 score_type,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="max_att"):
        """initialize max-attention layer"""       
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.unit_dim = unit_dim
        self.score_type = score_type
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.attention_matrix = _create_attention_matrix(self.src_dim,
                self.trg_dim, self.unit_dim, self.score_type)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call max-attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_data = input_src_data * input_src_mask
            input_trg_data = input_trg_data * input_trg_mask
            input_attention_score = _generate_attention_score(input_src_data,
                input_trg_data, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_mask, input_trg_mask, False)
            input_attention_score = tf.reduce_max(input_attention_score, axis=-2, keep_dims=True)
            input_attention_mask = tf.reduce_max(input_attention_mask, axis=-2, keep_dims=True)
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1, keepdims=True)
            output_attention = tf.matmul(input_attention_weight, input_trg_data)
            trg_max_length = tf.shape(input_trg_data)[1]
            output_attention = tf.tile(output_attention, multiples=[1, trg_max_length, 1])
            output_mask = input_trg_mask
        
        return output_attention, output_mask

class SelfAttention(object):
    """self-attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 unit_dim,
                 score_type,
                 num_gpus=1,
                 default_gpu_id=0,
                 trainable=True,
                 scope="self_att"):
        """initialize self-attention layer"""
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.unit_dim = unit_dim
        self.score_type = score_type
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.attention_matrix = _create_attention_matrix(self.src_dim,
                self.trg_dim, self.unit_dim, self.score_type)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call self-attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_data = input_src_data * input_src_mask
            input_trg_data = input_trg_data * input_trg_mask
            input_attention_score = _generate_attention_score(input_src_data,
                input_trg_data, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_mask, input_trg_mask, True)
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1, keepdims=True)
            output_attention = tf.matmul(input_attention_weight, input_trg_data)
            output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def get_attention_matrix():
        return self.attention_matrix
