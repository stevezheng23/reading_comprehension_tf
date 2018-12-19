import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

from layer.basic import *

__all__ = ["Attention", "MaxAttention", "CoAttention", "GatedAttention", "HeadAttention", "MultiHeadAttention"]

def _create_attention_matrix(src_unit_dim,
                             trg_unit_dim,
                             attention_unit_dim,
                             attention_score_type,
                             regularizer,
                             random_seed,
                             trainable):
    """create attetnion matrix"""
    if attention_score_type == "dot":
        attention_matrix = []
    elif attention_score_type == "scaled_dot":
        attention_matrix = []
    elif attention_score_type == "linear":
        attention_matrix = _create_linear_attention_matrix(src_unit_dim, trg_unit_dim, regularizer, random_seed, trainable)
    elif attention_score_type == "bilinear":
        attention_matrix = _create_bilinear_attention_matrix(src_unit_dim, trg_unit_dim, regularizer, random_seed, trainable)
    elif attention_score_type == "nonlinear":
        attention_matrix = _create_nonlinear_attention_matrix(src_unit_dim,
            trg_unit_dim, attention_unit_dim, regularizer, random_seed, trainable)
    elif attention_score_type == "linear_plus":
        attention_matrix = _create_linear_plus_attention_matrix(src_unit_dim, trg_unit_dim, regularizer, random_seed, trainable)
    elif attention_score_type == "nonlinear_plus":
        attention_matrix = _create_nonlinear_plus_attention_matrix(src_unit_dim,
            trg_unit_dim, attention_unit_dim, regularizer, random_seed, trainable)
    else:
        raise ValueError("unsupported attention score type {0}".format(attention_score_type))
    
    return attention_matrix

def _create_linear_attention_matrix(src_unit_dim,
                                    trg_unit_dim,
                                    regularizer,
                                    random_seed,
                                    trainable):
    """create linear attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    
    linear_src_weight = tf.get_variable("linear_src_weight", shape=[1, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    linear_trg_weight = tf.get_variable("linear_trg_weight", shape=[1, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [linear_src_weight, linear_trg_weight]
    
    return attention_matrix

def _create_bilinear_attention_matrix(src_unit_dim,
                                      trg_unit_dim,
                                      regularizer,
                                      random_seed,
                                      trainable):
    """create bilinear attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    
    bilinear_weight = tf.get_variable("bilinear_weight", shape=[src_unit_dim, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [bilinear_weight]
    
    return attention_matrix

def _create_nonlinear_attention_matrix(src_unit_dim,
                                       trg_unit_dim,
                                       attention_unit_dim,
                                       regularizer,
                                       random_seed,
                                       trainable):
    """create nonlinear attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    bias_initializer = create_variable_initializer("zero")
    
    pre_nonlinear_src_weight = tf.get_variable("pre_nonlinear_src_weight", shape=[attention_unit_dim, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_trg_weight = tf.get_variable("pre_nonlinear_trg_weight", shape=[attention_unit_dim, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_bias = tf.get_variable("pre_nonlinear_bias", shape=[attention_unit_dim],
        initializer=bias_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    post_nonlinear_weight = tf.get_variable("post_nonlinear_weight", shape=[1, attention_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [pre_nonlinear_src_weight, pre_nonlinear_trg_weight, pre_nonlinear_bias, post_nonlinear_weight]
    
    return attention_matrix

def _create_linear_plus_attention_matrix(src_unit_dim,
                                         trg_unit_dim,
                                         regularizer,
                                         random_seed,
                                         trainable):
    """create linear plus attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    
    if src_unit_dim != trg_unit_dim:
        raise ValueError("src dim {0} and trg dim must be the same for linear plus attention".format(src_unit_dim, trg_unit_dim))
    else:
        mul_unit_dim = src_unit_dim
    
    linear_plus_src_weight = tf.get_variable("linear_plus_src_weight", shape=[1, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    linear_plus_trg_weight = tf.get_variable("linear_plus_trg_weight", shape=[1, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    linear_plus_mul_weight = tf.get_variable("linear_plus_mul_weight", shape=[1, mul_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [linear_plus_src_weight, linear_plus_trg_weight, linear_plus_mul_weight]
    
    return attention_matrix

def _create_nonlinear_plus_attention_matrix(src_unit_dim,
                                            trg_unit_dim,
                                            attention_unit_dim,
                                            regularizer,
                                            random_seed,
                                            trainable):
    """create nonlinear plus attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    bias_initializer = create_variable_initializer("zero")
    
    if src_unit_dim != trg_unit_dim:
        raise ValueError("src dim {0} and trg dim must be the same for nonlinear plus attention".format(src_unit_dim, trg_unit_dim))
    else:
        mul_unit_dim = src_unit_dim
    
    pre_nonlinear_plus_src_weight = tf.get_variable("pre_nonlinear_plus_src_weight", shape=[attention_unit_dim, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_plus_trg_weight = tf.get_variable("pre_nonlinear_plus_trg_weight", shape=[attention_unit_dim, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_plus_mul_weight = tf.get_variable("pre_nonlinear_plus_mul_weight", shape=[attention_unit_dim, mul_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_plus_bias = tf.get_variable("pre_nonlinear_plus_bias", shape=[attention_unit_dim],
        initializer=bias_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    post_nonlinear_plus_weight = tf.get_variable("post_nonlinear_plus_weight", shape=[1, attention_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [pre_nonlinear_plus_src_weight, pre_nonlinear_plus_trg_weight,
        pre_nonlinear_plus_mul_weight, pre_nonlinear_plus_bias, post_nonlinear_plus_weight]
    
    return attention_matrix

def _create_projection_matrix(input_unit_dim,
                              projection_unit_dim,
                              regularizer,
                              random_seed,
                              trainable):
    """create projection matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    projection_weight = tf.get_variable("projection_weight", shape=[input_unit_dim, projection_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    
    return projection_weight

def _generate_attention_score(input_src_data,
                              input_trg_data,
                              attention_matrix,
                              attention_score_type):
    """generate attention score"""
    if attention_score_type == "dot":
        input_attention_score = _generate_dot_attention_score(input_src_data, input_trg_data)
    elif attention_score_type == "scaled_dot":
        input_attention_score = _generate_scaled_dot_attention_score(input_src_data, input_trg_data)
    elif attention_score_type == "linear":
        input_attention_score = _generate_linear_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    elif attention_score_type == "bilinear":
        input_attention_score = _generate_bilinear_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    elif attention_score_type == "nonlinear":
        input_attention_score = _generate_nonlinear_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    elif attention_score_type == "linear_plus":
        input_attention_score = _generate_linear_plus_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    elif attention_score_type == "nonlinear_plus":
        input_attention_score = _generate_nonlinear_plus_attention_score(input_src_data,
            input_trg_data, attention_matrix)
    else:
        raise ValueError("unsupported attention score type {0}".format(attention_score_type))
    
    return input_attention_score

def _generate_dot_attention_score(input_src_data,
                                  input_trg_data):
    """generate dot-product attention score"""
    input_attention = tf.matmul(input_src_data, input_trg_data, transpose_b=True)
    
    return input_attention

def _generate_scaled_dot_attention_score(input_src_data,
                                         input_trg_data):
    """generate scaled dot-product attention score"""
    src_unit_dim = tf.shape(input_src_data)[2]
    input_attention = tf.matmul(input_src_data, input_trg_data, transpose_b=True)
    input_attention = input_attention / tf.sqrt(tf.cast(src_unit_dim, dtype=tf.float32))
    
    return input_attention

def _generate_linear_attention_score(input_src_data,
                                     input_trg_data,
                                     attention_matrix):
    """generate linear attention score"""
    input_src_shape = tf.shape(input_src_data)
    input_trg_shape = tf.shape(input_trg_data)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    src_unit_dim = input_src_shape[2]
    trg_unit_dim = input_trg_shape[2]
    linear_src_weight = attention_matrix[0]
    linear_trg_weight = attention_matrix[1]
    input_src_data = tf.reshape(input_src_data, shape=[-1, src_unit_dim])
    input_src_data = tf.matmul(input_src_data, linear_src_weight, transpose_b=True)
    input_src_data = tf.reshape(input_src_data, shape=[batch_size, src_max_length, 1, -1])
    input_trg_data = tf.reshape(input_trg_data, shape=[-1, trg_unit_dim])
    input_trg_data = tf.matmul(input_trg_data, linear_trg_weight, transpose_b=True)
    input_trg_data = tf.reshape(input_trg_data, shape=[batch_size, 1, trg_max_length, -1])
    input_src_data = tf.tile(input_src_data, multiples=[1, 1, trg_max_length, 1])
    input_trg_data = tf.tile(input_trg_data, multiples=[1, src_max_length, 1, 1])
    input_attention = input_src_data + input_trg_data
    input_attention = tf.reshape(input_attention, shape=[batch_size, src_max_length, trg_max_length])
    
    return input_attention

def _generate_bilinear_attention_score(input_src_data,
                                       input_trg_data,
                                       attention_matrix):
    """generate bilinear attention score"""
    input_src_shape = tf.shape(input_src_data)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    src_unit_dim = input_src_shape[2]
    bilinear_weight = attention_matrix[0]
    input_src_data = tf.reshape(input_src_data, shape=[-1, src_unit_dim])
    input_src_data = tf.matmul(input_src_data, bilinear_weight)
    input_src_data = tf.reshape(input_src_data, shape=[batch_size, src_max_length, -1])
    input_attention = tf.matmul(input_src_data, input_trg_data, transpose_b=True)
    
    return input_attention

def _generate_nonlinear_attention_score(input_src_data,
                                        input_trg_data,
                                        attention_matrix):
    """generate linear attention score"""
    input_src_shape = tf.shape(input_src_data)
    input_trg_shape = tf.shape(input_trg_data)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    src_unit_dim = input_src_shape[2]
    trg_unit_dim = input_trg_shape[2]
    pre_nonlinear_src_weight = attention_matrix[0]
    pre_nonlinear_trg_weight = attention_matrix[1]
    pre_nonlinear_bias = tf.reshape(attention_matrix[2], shape=[1, 1, 1, -1])
    post_nonlinear_weight = attention_matrix[3]
    input_src_data = tf.reshape(input_src_data, shape=[-1, src_unit_dim])
    input_src_data = tf.matmul(input_src_data, pre_nonlinear_src_weight, transpose_b=True)
    input_src_data = tf.reshape(input_src_data, shape=[batch_size, src_max_length, 1, -1])
    input_trg_data = tf.reshape(input_trg_data, shape=[-1, trg_unit_dim])
    input_trg_data = tf.matmul(input_trg_data, pre_nonlinear_trg_weight, transpose_b=True)
    input_trg_data = tf.reshape(input_trg_data, shape=[batch_size, 1, trg_max_length, -1])
    input_src_data = tf.tile(input_src_data, multiples=[1, 1, trg_max_length, 1])
    input_trg_data = tf.tile(input_trg_data, multiples=[1, src_max_length, 1, 1])
    input_attention = input_src_data + input_trg_data
    input_attention = tf.nn.tanh(input_attention + pre_nonlinear_bias)
    attention_dim = tf.shape(input_attention)[-1]
    input_attention = tf.reshape(input_attention, shape=[-1, attention_dim])
    input_attention = tf.matmul(input_attention, post_nonlinear_weight, transpose_b=True)
    input_attention = tf.reshape(input_attention, shape=[batch_size, src_max_length, trg_max_length])
    
    return input_attention

def _generate_linear_plus_attention_score(input_src_data,
                                          input_trg_data,
                                          attention_matrix):
    """generate linear plus attention score"""
    input_src_shape = tf.shape(input_src_data)
    input_trg_shape = tf.shape(input_trg_data)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    src_unit_dim = input_src_shape[2]
    trg_unit_dim = input_trg_shape[2]
    mul_unit_dim = src_unit_dim
    linear_plus_src_weight = attention_matrix[0]
    linear_plus_trg_weight = attention_matrix[1]
    linear_plus_mul_weight = attention_matrix[2]
    input_src_data = tf.expand_dims(input_src_data, axis=2)
    input_trg_data = tf.expand_dims(input_trg_data, axis=1)
    input_src_data = tf.tile(input_src_data, multiples=[1, 1, trg_max_length, 1])
    input_trg_data = tf.tile(input_trg_data, multiples=[1, src_max_length, 1, 1])
    input_mul_data = input_src_data * input_trg_data
    input_src_data = tf.reshape(input_src_data, shape=[-1, src_unit_dim])
    input_src_data = tf.matmul(input_src_data, linear_plus_src_weight, transpose_b=True)
    input_trg_data = tf.reshape(input_trg_data, shape=[-1, trg_unit_dim])
    input_trg_data = tf.matmul(input_trg_data, linear_plus_trg_weight, transpose_b=True)
    input_mul_data = tf.reshape(input_mul_data, shape=[-1, mul_unit_dim])
    input_mul_data = tf.matmul(input_mul_data, linear_plus_mul_weight, transpose_b=True)
    input_attention = input_src_data + input_trg_data + input_mul_data
    input_attention = tf.reshape(input_attention, shape=[batch_size, src_max_length, trg_max_length])
    
    return input_attention

def _generate_nonlinear_plus_attention_score(input_src_data,
                                             input_trg_data,
                                             attention_matrix):
    """generate nonlinear plus attention score"""
    input_src_shape = tf.shape(input_src_data)
    input_trg_shape = tf.shape(input_trg_data)
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    src_unit_dim = input_src_shape[2]
    trg_unit_dim = input_trg_shape[2]
    mul_unit_dim = src_unit_dim
    pre_nonlinear_plus_src_weight = attention_matrix[0]
    pre_nonlinear_plus_trg_weight = attention_matrix[1]
    pre_nonlinear_plus_mul_weight = attention_matrix[2]
    pre_nonlinear_plus_bias = tf.reshape(attention_matrix[3], shape=[1, 1, 1, -1])
    post_nonlinear_plus_weight = attention_matrix[4]
    input_src_data = tf.reshape(input_src_data, shape=[batch_size, src_max_length, 1, -1])
    input_trg_data = tf.reshape(input_trg_data, shape=[batch_size, 1, trg_max_length, -1])
    input_src_data = tf.tile(input_src_data, multiples=[1, 1, trg_max_length, 1])
    input_trg_data = tf.tile(input_trg_data, multiples=[1, src_max_length, 1, 1])
    input_mul_data = input_src_data * input_trg_data
    input_src_data = tf.reshape(input_src_data, shape=[-1, src_unit_dim])
    input_src_data = tf.matmul(input_src_data, pre_nonlinear_plus_src_weight, transpose_b=True)
    input_trg_data = tf.reshape(input_trg_data, shape=[-1, trg_unit_dim])
    input_trg_data = tf.matmul(input_trg_data, pre_nonlinear_plus_trg_weight, transpose_b=True)
    input_mul_data = tf.reshape(input_mul_data, shape=[-1, mul_unit_dim])
    input_mul_data = tf.matmul(input_mul_data, pre_nonlinear_plus_mul_weight, transpose_b=True)
    input_attention = input_src_data + input_trg_data + input_mul_data
    input_attention = tf.nn.tanh(input_attention + pre_nonlinear_plus_bias)
    input_attention = tf.matmul(input_attention, post_nonlinear_plus_weight, transpose_b=True)
    input_attention = tf.reshape(input_attention, shape=[batch_size, src_max_length, trg_max_length])
    
    return input_attention

def _generate_attention_mask(input_src_mask,
                             input_trg_mask,
                             remove_diag=False):
    """generate attention mask"""
    input_mask = tf.matmul(input_src_mask, input_trg_mask, transpose_b=True)
    
    if remove_diag == True:
        src_max_length = tf.shape(input_src_mask)[1]
        trg_max_length = tf.shape(input_trg_mask)[1]
        input_mask = input_mask * (1 - tf.eye(src_max_length, trg_max_length))
    
    return input_mask

def _generate_projection_data(input_data,
                              projection_matrix):
    """generate projection data"""
    input_shape = tf.shape(input_data)
    batch_size = input_shape[0]
    max_length = input_shape[1]
    unit_dim = input_shape[2]
    input_projection = tf.reshape(input_data, shape=[-1, unit_dim])
    input_projection = tf.matmul(input_projection, projection_matrix)
    input_projection = tf.reshape(input_projection, shape=[batch_size, max_length, -1])
    
    return input_projection

class Attention(object):
    """attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 is_self=False,
                 external_matrix=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="attention"):
        """initialize attention layer"""
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.att_dim = att_dim
        self.score_type = score_type
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.is_self = is_self
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if external_matrix == None:
                self.attention_matrix = _create_attention_matrix(self.src_dim, self.trg_dim,
                    self.att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable)
            else:
                self.attention_matrix = external_matrix
            
            if self.layer_norm == True:
                self.src_norm_layer = LayerNorm(layer_dim=self.src_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="src_layer_norm")
                
                if self.is_self == True:
                    self.trg_norm_layer = self.src_norm_layer
                else:
                    self.trg_norm_layer = LayerNorm(layer_dim=self.trg_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="trg_layer_norm")
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_attention = input_src_data
            input_trg_attention = input_trg_data
            input_src_attention_mask = input_src_mask
            input_trg_attention_mask = input_trg_mask
            
            if self.layer_norm == True:
                input_src_attention, input_src_attention_mask = self.src_norm_layer(input_src_attention, input_src_attention_mask)
                input_trg_attention, input_trg_attention_mask = self.trg_norm_layer(input_trg_attention, input_trg_attention_mask)
            
            input_attention_score = _generate_attention_score(input_src_attention,
                input_trg_attention, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_attention_mask, input_trg_attention_mask, self.is_self)
            output_attention_score = input_attention_score
            output_score_mask = input_attention_mask
            
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1, keepdims=True) * input_attention_mask
            input_attention = tf.matmul(input_attention_weight, input_trg_attention)
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
            else:
                output_attention = input_attention
                output_mask = input_src_mask
        
        return output_attention, output_mask, output_attention_score, output_score_mask
    
    def get_attention_matrix(self):
        return self.attention_matrix

class MaxAttention(object):
    """max-attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 is_self=False,
                 external_matrix=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="max_att"):
        """initialize max-attention layer"""       
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.att_dim = att_dim
        self.score_type = score_type
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.is_self = is_self
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if external_matrix == None:
                self.attention_matrix = _create_attention_matrix(self.src_dim, self.trg_dim,
                    self.att_dim, self.regularizer, self.score_type, self.random_seed, self.trainable)
            else:
                self.attention_matrix = external_matrix
            
            if self.layer_norm == True:
                self.src_norm_layer = LayerNorm(layer_dim=self.src_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="src_layer_norm")
                
                if self.is_self == True:
                    self.trg_norm_layer = self.src_norm_layer
                else:
                    self.trg_norm_layer = LayerNorm(layer_dim=self.trg_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="trg_layer_norm")
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call max-attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_attention = input_src_data
            input_trg_attention = input_trg_data
            input_src_attention_mask = input_src_mask
            input_trg_attention_mask = input_trg_mask
            
            if self.layer_norm == True:
                input_src_attention, input_src_attention_mask = self.src_norm_layer(input_src_attention, input_src_attention_mask)
                input_trg_attention, input_trg_attention_mask = self.trg_norm_layer(input_trg_attention, input_trg_attention_mask)
            
            input_attention_score = _generate_attention_score(input_src_attention,
                input_trg_attention, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_attention_mask, input_trg_attention_mask, self.is_self)
            input_attention_score = tf.reduce_max(input_attention_score, axis=-1, keepdims=True)
            input_attention_mask = tf.reduce_max(input_attention_mask, axis=-1, keepdims=True)
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=1, keepdims=True) * input_attention_mask
            input_attention_weight = tf.transpose(input_attention_weight, perm=[0, 2, 1])
            input_attention = tf.matmul(input_attention_weight, input_src_attention)
            src_max_length = tf.shape(input_src_attention)[1]
            input_attention = tf.tile(input_attention, multiples=[1, src_max_length, 1])
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
            else:
                output_attention = input_attention
                output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def get_attention_matrix(self):
        return self.attention_matrix

class CoAttention(object):
    """co-attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 is_self=False,
                 external_matrix=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="co_att"):
        """initialize co-attention layer"""       
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.att_dim = att_dim
        self.score_type = score_type
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.is_self = is_self
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if external_matrix == None:
                self.attention_matrix = _create_attention_matrix(self.src_dim, self.trg_dim,
                    self.att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable)
            else:
                self.attention_matrix = external_matrix
            
            if self.layer_norm == True:
                self.src_norm_layer = LayerNorm(layer_dim=self.src_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="src_layer_norm")
                
                if self.is_self == True:
                    self.trg_norm_layer = self.src_norm_layer
                else:
                    self.trg_norm_layer = LayerNorm(layer_dim=self.trg_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="trg_layer_norm")
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call co-attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_attention = input_src_data
            input_trg_attention = input_trg_data
            input_src_attention_mask = input_src_mask
            input_trg_attention_mask = input_trg_mask
            
            if self.layer_norm == True:
                input_src_attention, input_src_attention_mask = self.src_norm_layer(input_src_attention, input_src_attention_mask)
                input_trg_attention, input_trg_attention_mask = self.trg_norm_layer(input_trg_attention, input_trg_attention_mask)
            
            input_attention_score = _generate_attention_score(input_src_attention,
                input_trg_attention, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_attention_mask, input_trg_attention_mask, self.is_self)
            input_attention_s2t_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1, keepdims=True) * input_attention_mask
            input_attention_t2s_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=1, keepdims=True) * input_attention_mask
            input_attention_t2s_weight = tf.transpose(input_attention_t2s_weight, perm=[0, 2, 1])
            input_attention = tf.matmul(input_attention_t2s_weight, input_src_attention)
            input_attention = tf.matmul(input_attention_s2t_weight, input_attention)
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
            else:
                output_attention = input_attention
                output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def get_attention_matrix(self):
        return self.attention_matrix

class GatedAttention(object):
    """gated-attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 is_self=False,
                 external_matrix=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="gated_att"):
        """initialize gated-attention layer"""
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.att_dim = att_dim
        self.score_type = score_type
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.is_self = is_self
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if external_matrix == None:
                self.attention_matrix = _create_attention_matrix(self.src_dim, self.trg_dim,
                    self.att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable)
            else:
                self.attention_matrix = external_matrix
            
            if self.layer_norm == True:
                self.src_norm_layer = LayerNorm(layer_dim=self.src_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="src_layer_norm")
                
                if self.is_self == True:
                    self.trg_norm_layer = self.src_norm_layer
                else:
                    self.trg_norm_layer = LayerNorm(layer_dim=self.trg_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="trg_layer_norm")
            
            weight_initializer = create_variable_initializer("glorot_uniform")
            gate_activation = create_activation_function("sigmoid")
            if self.residual_connect == True and self.is_self == True:
                self.gate_layer = tf.layers.Dense(units=self.trg_dim, activation=gate_activation,
                    kernel_initializer=weight_initializer, kernel_regularizer=self.regularizer, trainable=self.trainable)
            else:
                self.gate_layer = tf.layers.Dense(units=self.src_dim+self.trg_dim, activation=gate_activation,
                    kernel_initializer=weight_initializer, kernel_regularizer=self.regularizer, trainable=self.trainable)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call gated-attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_attention = input_src_data
            input_trg_attention = input_trg_data
            input_src_attention_mask = input_src_mask
            input_trg_attention_mask = input_trg_mask
            
            if self.layer_norm == True:
                input_src_attention, input_src_attention_mask = self.src_norm_layer(input_src_attention, input_src_attention_mask)
                input_trg_attention, input_trg_attention_mask = self.trg_norm_layer(input_trg_attention, input_trg_attention_mask)
            
            input_attention_score = _generate_attention_score(input_src_attention,
                input_trg_attention, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_attention_mask, input_trg_attention_mask, self.is_self)
            
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1, keepdims=True) * input_attention_mask
            input_attention = tf.matmul(input_attention_weight, input_trg_attention)
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (self.gate_layer(input_attention) * input_attention + input_src_data, input_src_mask))
            else:
                input_attention = tf.concat([input_src_data, input_attention], axis=-1) 
                gate = self.gate_layer(input_attention)
                output_attention = gate * input_attention
                output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def get_attention_matrix(self):
        return self.attention_matrix

class HeadAttention(object):
    """head-attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 layer_norm=False,
                 is_self=False,
                 external_matrix=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="head_att"):
        """initialize head-attention layer"""
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.att_dim = att_dim
        self.score_type = score_type
        self.layer_norm = layer_norm
        self.is_self = is_self
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if external_matrix == None:
                (q_att_dim, k_att_dim, v_att_dim) = tuple(self.att_dim)
                self.projection_matrix = [
                    _create_projection_matrix(self.src_dim, q_att_dim, self.regularizer, self.random_seed, self.trainable),
                    _create_projection_matrix(self.trg_dim, k_att_dim, self.regularizer, self.random_seed, self.trainable),
                    _create_projection_matrix(self.trg_dim, v_att_dim, self.regularizer, self.random_seed, self.trainable)
                ]
                self.attention_matrix = _create_attention_matrix(q_att_dim, k_att_dim,
                    k_att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable)
            else:
                self.projection_matrix = external_matrix["projection"]
                self.attention_matrix = external_matrix["attention"]
            
            if self.layer_norm == True:
                self.src_norm_layer = LayerNorm(layer_dim=self.src_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="src_layer_norm")
                
                if self.is_self == True:
                    self.trg_norm_layer = self.src_norm_layer
                else:
                    self.trg_norm_layer = LayerNorm(layer_dim=self.trg_dim, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
                    regularizer=self.regularizer, trainable=self.trainable, scope="trg_layer_norm")
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call head-attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_attention = input_src_data
            input_trg_attention = input_trg_data
            input_src_attention_mask = input_src_mask
            input_trg_attention_mask = input_trg_mask
            
            if self.layer_norm == True:
                input_src_attention, input_src_attention_mask = self.src_norm_layer(input_src_attention, input_src_attention_mask)
                input_trg_attention, input_trg_attention_mask = self.trg_norm_layer(input_trg_attention, input_trg_attention_mask)
            
            input_query_attention = _generate_projection_data(input_src_attention, self.projection_matrix[0])
            input_key_attention = _generate_projection_data(input_trg_attention, self.projection_matrix[1])
            input_value_attention = _generate_projection_data(input_trg_attention, self.projection_matrix[2])
            input_attention_score = _generate_attention_score(input_query_attention,
                input_key_attention, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_src_attention_mask, input_trg_attention_mask, self.is_self)
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1, keepdims=True) * input_attention_mask
            input_attention = tf.matmul(input_attention_weight, input_value_attention)
            
            output_attention = input_attention
            output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def get_projection_matrix(self):
        return self.projection_matrix
    
    def get_attention_matrix(self):
        return self.attention_matrix

class MultiHeadAttention(object):
    """multi-head attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 is_self=False,
                 external_matrix=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="multi_head_att"):
        """initialize multi-head attention layer"""
        self.src_dim = src_dim
        self.trg_dim = trg_dim
        self.att_dim = att_dim
        self.score_type = score_type
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.is_self = is_self
        self.external_matrix=external_matrix
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.attention_layer_list = []
            for i in range(len(self.att_dim)):
                layer_scope = "head_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id
                attention_layer = HeadAttention(src_dim=self.src_dim, trg_dim=self.trg_dim,
                    att_dim=self.att_dim[i], score_type=self.score_type, layer_norm=self.layer_norm, is_self=self.is_self,
                    external_matrix=self.external_matrix, num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id,
                    regularizer=self.regularizer,random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.attention_layer_list.append(attention_layer)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call multi-head attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_attention_list = []
            input_attention_mask_list = []
            for attention_layer in self.attention_layer_list:
                input_attention, input_attention_mask = attention_layer(input_src_data,
                    input_trg_data, input_src_mask, input_trg_mask)
                input_attention_list.append(input_attention)
                input_attention_mask_list.append(input_attention_mask)
            
            input_attention = tf.concat(input_attention_list, axis=-1)
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
            else:
                output_attention = input_attention
                output_mask = input_src_mask
        
        return output_attention, output_mask
