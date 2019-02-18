import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *

from layer.basic import *

__all__ = ["Attention", "MaxAttention", "CoAttention", "GatedAttention", "MultiHeadAttention"]

def _create_attention_matrix(src_unit_dim,
                             trg_unit_dim,
                             attention_unit_dim,
                             attention_score_type,
                             regularizer,
                             random_seed,
                             trainable,
                             scope="att_matrix"):
    """create attetnion matrix"""
    scope = "{0}/{1}".format(scope, attention_score_type)
    if attention_score_type == "dot":
        attention_matrix = []
    elif attention_score_type == "scaled_dot":
        attention_matrix = []
    elif attention_score_type == "linear":
        attention_matrix = _create_linear_attention_matrix(src_unit_dim,
            trg_unit_dim, regularizer, random_seed, trainable, scope)
    elif attention_score_type == "bilinear":
        attention_matrix = _create_bilinear_attention_matrix(src_unit_dim,
            trg_unit_dim, regularizer, random_seed, trainable, scope)
    elif attention_score_type == "nonlinear":
        attention_matrix = _create_nonlinear_attention_matrix(src_unit_dim,
            trg_unit_dim, attention_unit_dim, regularizer, random_seed, trainable, scope)
    elif attention_score_type == "linear_plus":
        attention_matrix = _create_linear_plus_attention_matrix(src_unit_dim,
            trg_unit_dim, regularizer, random_seed, trainable, scope)
    elif attention_score_type == "nonlinear_plus":
        attention_matrix = _create_nonlinear_plus_attention_matrix(src_unit_dim,
            trg_unit_dim, attention_unit_dim, regularizer, random_seed, trainable, scope)
    elif attention_score_type == "trilinear":
        attention_matrix = _create_trilinear_attention_matrix(src_unit_dim,
            trg_unit_dim, regularizer, random_seed, trainable, scope)
    else:
        raise ValueError("unsupported attention score type {0}".format(attention_score_type))
    
    return attention_matrix

def _create_linear_attention_matrix(src_unit_dim,
                                    trg_unit_dim,
                                    regularizer,
                                    random_seed,
                                    trainable,
                                    scope="linear"):
    """create linear attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    
    linear_src_weight = tf.get_variable("{0}/src_weight".format(scope), shape=[1, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    linear_trg_weight = tf.get_variable("{0}/trg_weight".format(scope), shape=[1, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [linear_src_weight, linear_trg_weight]
    
    return attention_matrix

def _create_bilinear_attention_matrix(src_unit_dim,
                                      trg_unit_dim,
                                      regularizer,
                                      random_seed,
                                      trainable,
                                      scope="bilinear"):
    """create bilinear attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    
    bilinear_weight = tf.get_variable("{0}/weight".format(scope), shape=[src_unit_dim, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [bilinear_weight]
    
    return attention_matrix

def _create_nonlinear_attention_matrix(src_unit_dim,
                                       trg_unit_dim,
                                       attention_unit_dim,
                                       regularizer,
                                       random_seed,
                                       trainable,
                                       scope="nonlinear"):
    """create nonlinear attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    bias_initializer = create_variable_initializer("zero")
    
    pre_nonlinear_src_weight = tf.get_variable("{0}/pre/src_weight".format(scope), shape=[attention_unit_dim, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_trg_weight = tf.get_variable("{0}/pre/trg_weight".format(scope), shape=[attention_unit_dim, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_bias = tf.get_variable("{0}/pre/bias".format(scope), shape=[attention_unit_dim],
        initializer=bias_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    post_nonlinear_weight = tf.get_variable("{0}/post/weight".format(scope), shape=[1, attention_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [pre_nonlinear_src_weight, pre_nonlinear_trg_weight, pre_nonlinear_bias, post_nonlinear_weight]
    
    return attention_matrix

def _create_linear_plus_attention_matrix(src_unit_dim,
                                         trg_unit_dim,
                                         regularizer,
                                         random_seed,
                                         trainable,
                                         scope="linear_plus"):
    """create linear plus attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    
    if src_unit_dim != trg_unit_dim:
        raise ValueError("src dim {0} and trg dim must be the same for linear plus attention".format(src_unit_dim, trg_unit_dim))
    else:
        mul_unit_dim = src_unit_dim
    
    linear_plus_src_weight = tf.get_variable("{0}/src_weight".format(scope), shape=[1, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    linear_plus_trg_weight = tf.get_variable("{0}/trg_weight".format(scope), shape=[1, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    linear_plus_mul_weight = tf.get_variable("{0}/mul_weight".format(scope), shape=[1, mul_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [linear_plus_src_weight, linear_plus_trg_weight, linear_plus_mul_weight]
    
    return attention_matrix

def _create_nonlinear_plus_attention_matrix(src_unit_dim,
                                            trg_unit_dim,
                                            attention_unit_dim,
                                            regularizer,
                                            random_seed,
                                            trainable,
                                            scope="nonlinear_plus"):
    """create nonlinear plus attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    bias_initializer = create_variable_initializer("zero")
    
    if src_unit_dim != trg_unit_dim:
        raise ValueError("src dim {0} and trg dim must be the same for nonlinear plus attention".format(src_unit_dim, trg_unit_dim))
    else:
        mul_unit_dim = src_unit_dim
    
    pre_nonlinear_plus_src_weight = tf.get_variable("{0}/pre/src_weight".format(scope), shape=[attention_unit_dim, src_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_plus_trg_weight = tf.get_variable("{0}/pre/trg_weight".format(scope), shape=[attention_unit_dim, trg_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_plus_mul_weight = tf.get_variable("{0}/pre/mul_weight".format(scope), shape=[attention_unit_dim, mul_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    pre_nonlinear_plus_bias = tf.get_variable("{0}/pre/bias".format(scope), shape=[attention_unit_dim],
        initializer=bias_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    post_nonlinear_plus_weight = tf.get_variable("{0}/post/weight".format(scope), shape=[1, attention_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [pre_nonlinear_plus_src_weight, pre_nonlinear_plus_trg_weight,
        pre_nonlinear_plus_mul_weight, pre_nonlinear_plus_bias, post_nonlinear_plus_weight]
    
    return attention_matrix

def _create_trilinear_attention_matrix(src_unit_dim,
                                       trg_unit_dim,
                                       regularizer,
                                       random_seed,
                                       trainable,
                                       scope="trilinear"):
    """create trilinear attetnion matrix"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    
    if src_unit_dim != trg_unit_dim:
        raise ValueError("src dim {0} and trg dim must be the same for trilinear attention".format(src_unit_dim, trg_unit_dim))
    else:
        mul_unit_dim = src_unit_dim
    
    trilinear_src_weight = tf.get_variable("{0}/src_weight".format(scope), shape=[src_unit_dim, 1],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    trilinear_trg_weight = tf.get_variable("{0}/trg_weight".format(scope), shape=[trg_unit_dim, 1],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    trilinear_mul_weight = tf.get_variable("{0}/mul_weight".format(scope), shape=[1, 1, mul_unit_dim],
        initializer=weight_initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
    attention_matrix = [trilinear_src_weight, trilinear_trg_weight, trilinear_mul_weight]
    
    return attention_matrix

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
    elif attention_score_type == "trilinear":
        input_attention_score = _generate_trilinear_attention_score(input_src_data,
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

def _generate_trilinear_attention_score(input_src_data,
                                        input_trg_data,
                                        attention_matrix):
    """generate trilinear attention score"""
    input_src_shape = tf.shape(input_src_data) # [batch_size, src_len, d]
    input_trg_shape = tf.shape(input_trg_data) # [batch_size, trg_len, d]
    batch_size = input_src_shape[0]
    src_max_length = input_src_shape[1]
    trg_max_length = input_trg_shape[1]
    src_unit_dim = input_src_shape[2]
    trg_unit_dim = input_trg_shape[2]
    mul_unit_dim = src_unit_dim
    trilinear_src_weight = attention_matrix[0] # [d, 1]
    trilinear_trg_weight = attention_matrix[1] # [d, 1]
    trilinear_mul_weight = attention_matrix[2] # [1, 1, d]
    
    input_src_part = tf.reshape(input_src_data, shape=[-1, src_unit_dim]) # [-1, d]
    input_trg_part = tf.reshape(input_trg_data, shape=[-1, trg_unit_dim]) # [-1, d]
    input_src_part = tf.matmul(input_src_part, trilinear_src_weight) # [-1, 1]
    input_trg_part = tf.matmul(input_trg_part, trilinear_trg_weight) # [-1, 1]
    input_src_part = tf.reshape(input_src_part, shape=[batch_size, src_max_length, 1]) # [batch_size, src_len, 1]
    input_trg_part = tf.reshape(input_trg_part, shape=[batch_size, 1, trg_max_length]) # [batch_size, 1, trg_len]
    input_src_score = tf.tile(input_src_part, multiples=[1, 1, trg_max_length]) # [batch_size, src_len, trg_len]
    input_trg_score = tf.tile(input_trg_part, multiples=[1, src_max_length, 1]) # [batch_size, src_len, trg_len]
    
    input_src_part = input_src_data * trilinear_mul_weight # [batch_size, src_len, d]
    input_trg_part = tf.transpose(input_trg_data, perm=[0, 2, 1]) # [batch_size, d, trg_len]
    input_mul_score = tf.matmul(input_src_part, input_trg_part) # [batch_size, src_len, trg_len]
    
    input_attention = input_src_score + input_trg_score + input_mul_score # [batch_size, src_len, trg_len]
    
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

def _create_projection_layer(unit_dim,
                             hidden_activation,
                             use_bias,
                             regularizer,
                             random_seed,
                             trainable,
                             name):
    """create projection layer"""
    weight_initializer = create_variable_initializer("glorot_uniform", random_seed)
    bias_initializer = create_variable_initializer("zero")
    projection_layer = tf.layers.Dense(units=unit_dim, activation=hidden_activation,
        use_bias=use_bias, kernel_initializer=weight_initializer, bias_initializer=bias_initializer,
        kernel_regularizer=regularizer, bias_regularizer=regularizer, trainable=trainable, name=name)
    
    return projection_layer

class Attention(object):
    """attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 dropout,
                 att_dropout=0.0,
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
        self.dropout = dropout
        self.att_dropout = att_dropout
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
                    self.att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable, "att_matrix")
            else:
                self.attention_matrix = external_matrix
            
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            self.att_dropout_layer = Dropout(rate=self.att_dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed, scope="att_dropout")
            
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
            input_attention_mask = _generate_attention_mask(input_src_attention_mask,
                input_trg_attention_mask, self.is_self)
            input_attention_score = input_attention_score * input_attention_mask
            
            output_attention_score = input_attention_score
            output_score_mask = input_attention_mask
            
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1) * input_attention_mask
            input_attention_weight, _ = self.att_dropout_layer(input_attention_weight, input_attention_mask)
            
            input_attention = tf.matmul(input_attention_weight, input_trg_attention)
            input_attention, _ = self.dropout_layer(input_attention, input_src_mask)
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
                output_attention = output_attention * output_mask
            else:
                output_attention = input_attention * input_src_mask
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
                 dropout,
                 att_dropout=0.0,
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
        self.dropout = dropout
        self.att_dropout = att_dropout
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
                    self.att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable, "att_matrix")
            else:
                self.attention_matrix = external_matrix
            
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            self.att_dropout_layer = Dropout(rate=self.att_dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed, scope="att_dropout")
            
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
            input_attention_mask = _generate_attention_mask(input_src_attention_mask,
                input_trg_attention_mask, self.is_self)
            input_attention_score = tf.transpose(tf.reduce_max(input_attention_score, axis=-1, keepdims=True), perm=[0, 2, 1])
            input_attention_mask = tf.transpose(tf.reduce_max(input_attention_mask, axis=-1, keepdims=True), perm=[0, 2, 1])
            input_attention_score = input_attention_score * input_attention_mask
            
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1) * input_attention_mask
            input_attention_weight, _ = self.att_dropout_layer(input_attention_weight, input_attention_mask)
            
            input_attention = tf.matmul(input_attention_weight, input_src_attention)
            input_attention, _ = self.dropout_layer(input_attention, input_src_mask)
            
            src_max_length = tf.shape(input_src_attention)[1]
            input_attention = tf.tile(input_attention, multiples=[1, src_max_length, 1])
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
                output_attention = output_attention * output_mask
            else:
                output_attention = input_attention * input_src_mask
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
                 dropout,
                 att_dropout=0.0,
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
        self.dropout = dropout
        self.att_dropout = att_dropout
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
                    self.att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable, "att_matrix")
            else:
                self.attention_matrix = external_matrix
                        
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            self.s2t_att_dropout_layer = Dropout(rate=self.att_dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed, scope="s2t_att_dropout")
            self.t2s_att_dropout_layer = Dropout(rate=self.att_dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed, scope="t2s_att_dropout")
            
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
            input_attention_mask = _generate_attention_mask(input_src_attention_mask,
                input_trg_attention_mask, self.is_self)
            
            input_s2t_att_score = input_attention_score
            input_s2t_att_mask = input_attention_mask
            input_s2t_att_score = input_s2t_att_score * input_s2t_att_mask
            input_t2s_att_score = tf.transpose(input_attention_score, perm=[0, 2, 1])
            input_t2s_att_mask = tf.transpose(input_attention_mask, perm=[0, 2, 1])
            input_t2s_att_score = input_t2s_att_score * input_t2s_att_mask
            
            input_s2t_att_weight = softmax_with_mask(input_s2t_att_score,
                input_s2t_att_mask, axis=-1) * input_s2t_att_mask
            input_s2t_att_weight, _ = self.s2t_att_dropout_layer(input_s2t_att_weight, input_s2t_att_mask)
            input_t2s_att_weight = softmax_with_mask(input_t2s_att_score,
                input_t2s_att_mask, axis=-1) * input_t2s_att_mask
            input_t2s_att_weight, _ = self.t2s_att_dropout_layer(input_t2s_att_weight, input_t2s_att_mask)
            
            input_attention_weight = tf.matmul(input_s2t_att_weight, input_t2s_att_weight)
            input_attention = tf.matmul(input_attention_weight, input_src_attention)
            input_attention, _ = self.dropout_layer(input_attention, input_src_mask)
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
                output_attention = output_attention * output_mask
            else:
                output_attention = input_attention * input_src_mask
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
                 dropout,
                 att_dropout=0.0,
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
        self.dropout = dropout
        self.att_dropout = att_dropout
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
                    self.att_dim, self.score_type, self.regularizer, self.random_seed, self.trainable, "att_matrix")
            else:
                self.attention_matrix = external_matrix
                        
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            self.att_dropout_layer = Dropout(rate=self.att_dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed, scope="att_dropout")
            
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
            input_attention_mask = _generate_attention_mask(input_src_attention_mask,
                input_trg_attention_mask, self.is_self)
            input_attention_score = input_attention_score * input_attention_mask
                        
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1) * input_attention_mask
            input_attention_weight, _ = self.att_dropout_layer(input_attention_weight, input_attention_mask)
            
            input_attention = tf.matmul(input_attention_weight, input_trg_attention)
            input_attention, _ = self.dropout_layer(input_attention, input_src_mask)
                        
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (self.gate_layer(input_attention) * input_attention + input_src_data, input_src_mask))
                output_attention = output_attention * output_mask
            else:
                input_attention = tf.concat([input_src_data, input_attention], axis=-1) 
                gate = self.gate_layer(input_attention)
                output_attention = gate * input_attention * input_src_mask
                output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def get_attention_matrix(self):
        return self.attention_matrix

class MultiHeadAttention(object):
    """multi-head attention layer"""
    def __init__(self,
                 src_dim,
                 trg_dim,
                 att_dim,
                 num_head,
                 score_type,
                 dropout,
                 att_dropout=0.0,
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
        self.num_head = num_head
        self.score_type = score_type
        self.dropout = dropout
        self.att_dropout = att_dropout
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
                query_dim = self.att_dim
                key_dim = self.att_dim
                value_dim = self.trg_dim
                self.projection_layer = {
                    "query": _create_projection_layer(query_dim, None, False,
                        self.regularizer, self.random_seed, self.trainable, "query_projection"),
                    "key": _create_projection_layer(key_dim, None, False,
                        self.regularizer, self.random_seed, self.trainable, "key_projection"),
                    "value": _create_projection_layer(value_dim, None, False,
                        self.regularizer, self.random_seed, self.trainable, "value_projection")
                }
                
                if self.att_dim % self.num_head != 0 or self.att_dim / self.num_head == 0:
                    raise ValueError("att dim {0} and # head {1} mis-match".format(self.att_dim, self.num_head))
                
                head_dim = self.att_dim / self.num_head
                self.attention_matrix = _create_attention_matrix(head_dim, head_dim,
                    head_dim, self.score_type, self.regularizer, self.random_seed, self.trainable, "att_matrix")
            else:
                self.projection_layer = external_matrix["projection"]
                self.attention_matrix = external_matrix["attention"]
                        
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            self.att_dropout_layer = Dropout(rate=self.att_dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed, scope="att_dropout")
            
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
        """call multi-head attention layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_src_shape = tf.shape(input_src_data)
            input_trg_shape = tf.shape(input_trg_data)
            input_src_attention = input_src_data
            input_trg_attention = input_trg_data
            input_src_attention_mask = input_src_mask
            input_trg_attention_mask = input_trg_mask
            
            if self.layer_norm == True:
                input_src_attention, input_src_attention_mask = self.src_norm_layer(input_src_attention, input_src_attention_mask)
                input_trg_attention, input_trg_attention_mask = self.trg_norm_layer(input_trg_attention, input_trg_attention_mask)
            
            input_query_attention = self.projection_layer["query"](input_src_attention)
            input_key_attention = self.projection_layer["key"](input_trg_attention)
            input_value_attention = self.projection_layer["value"](input_trg_attention)
            
            input_query_attention = self.__split_multi_head(input_query_attention,
                input_src_shape[0], input_src_shape[1], self.num_head)
            input_key_attention = self.__split_multi_head(input_key_attention,
                input_trg_shape[0], input_trg_shape[1], self.num_head)
            input_value_attention = self.__split_multi_head(input_value_attention,
                input_trg_shape[0], input_trg_shape[1], self.num_head)
            
            input_query_attention_mask = self.__split_multi_head_mask(input_src_attention_mask,
                input_src_shape[0], input_src_shape[1], self.num_head)
            input_key_attention_mask = self.__split_multi_head_mask(input_trg_attention_mask,
                input_trg_shape[0], input_trg_shape[1], self.num_head)
            
            input_attention_score = _generate_attention_score(input_query_attention,
                input_key_attention, self.attention_matrix, self.score_type)
            input_attention_mask = _generate_attention_mask(input_query_attention_mask,
                input_key_attention_mask, self.is_self)
            input_attention_score = input_attention_score * input_attention_mask
            
            input_attention_weight = softmax_with_mask(input_attention_score,
                input_attention_mask, axis=-1) * input_attention_mask
            input_attention_weight, _ = self.att_dropout_layer(input_attention_weight, input_attention_mask)
            
            input_attention = tf.matmul(input_attention_weight, input_value_attention)
            input_attention = self.__merge_multi_head(input_attention,
                input_src_shape[0], input_src_shape[1], self.num_head)
            input_attention, _ = self.dropout_layer(input_attention, input_src_mask)
            
            if self.residual_connect == True and self.is_self == True:
                output_attention, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_src_data, input_src_mask),
                    lambda: (input_attention + input_src_data, input_src_mask))
                output_attention = output_attention * output_mask
            else:
                output_attention = input_attention * input_src_mask
                output_mask = input_src_mask
        
        return output_attention, output_mask
    
    def __split_multi_head(self,
                           input_data,
                           batch_size,
                           max_length,
                           num_head):
        """split multi-head"""
        input_split_data = tf.reshape(input_data,
            shape=[batch_size, max_length, num_head, -1]) # [batch_size, max_len, num_head, -1]
        input_split_data = tf.transpose(input_split_data, perm=[0,2,1,3]) # [batch_size, num_head, max_len, -1]
        input_split_data = tf.reshape(input_split_data,
            shape=[batch_size * num_head, max_length, -1]) # [batch_size * num_head, max_len, -1]

        return input_split_data
    
    def __split_multi_head_mask(self,
                                input_mask,
                                batch_size,
                                max_length,
                                num_head):
        """split multi-head"""
        input_split_mask = tf.expand_dims(input_mask, axis=1) # [batch_size, 1, max_len, 1]
        input_split_mask = tf.tile(input_split_mask,
            multiples=[1, num_head, 1, 1]) # [batch_size, num_head, max_len, 1]
        input_split_mask = tf.reshape(input_split_mask,
            shape=[batch_size * num_head, max_length, 1]) # [batch_size * num_head, max_len, 1]

        return input_split_mask
    
    def __merge_multi_head(self,
                           input_data,
                           batch_size,
                           max_length,
                           num_head):
        """merge multi-head"""
        input_merge_data = tf.reshape(input_data,
            shape=[batch_size, num_head, max_length, -1]) # [batch_size, num_head, max_len, -1]
        input_merge_data = tf.transpose(input_merge_data, perm=[0,2,1,3]) # [batch_size, max_len, num_head, -1]
        input_merge_data = tf.reshape(input_merge_data,
            shape=[batch_size, max_length, -1]) # [batch_size, max_len, -1]

        return input_merge_data
    
    def get_projection_matrix(self):
        return self.projection_matrix
