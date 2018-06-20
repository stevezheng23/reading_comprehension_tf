import numpy as np
import tensorflow as tf

from layer.embedding import *
from layer.convolution import *
from layer.pooling import *
from layer.dense import *
from layer.highway import *
from layer.recurrent import *
from layer.attention import *

__all__ = ["create_embedding_layer", "create_convolution_layer", "create_pooling_layer",
           "create_dense_layer", "create_highway_layer", "create_recurrent_layer", "create_attention_layer"]

def create_embedding_layer(vocab_size,
                           embed_dim,
                           pretrained,
                           num_gpus,
                           default_gpu_id,
                           trainable):
    """create pooling layer"""
    if pretrained == True:
        embed_layer = PretrainedEmbedding(vocab_size=vocab_size, embed_dim=embed_dim,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable)
    else:
        embed_layer = Embedding(vocab_size=vocab_size, embed_dim=embed_dim,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable)
    
    return embed_layer

def create_convolution_layer(conv_type,
                             num_channel,
                             num_filter,
                             num_multiplier,
                             window_size,
                             stride_size,
                             padding_type,
                             activation,
                             dropout,
                             layer_norm,
                             residual_connect,
                             num_gpus,
                             default_gpu_id,
                             trainable):
    """create convolution layer"""
    scope = "conv/{0}".format(conv_type)
    if conv_type == "1d":
        conv_layer = Conv1D(num_channel=num_channel, num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, dropout=dropout, layer_norm=layer_norm, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif conv_type == "2d":
        conv_layer = Conv2D(num_channel=num_channel, num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, dropout=dropout, layer_norm=layer_norm, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif conv_type == "multi_1d":
        conv_layer = MultiConv1D(num_channel=num_channel, num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, dropout=dropout, layer_norm=layer_norm, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif conv_type == "multi_2d":
        conv_layer = MultiConv2D(num_channel=num_channel, num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, dropout=dropout, layer_norm=layer_norm, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif conv_type == "sep_1d":
        conv_layer = SeparableConv1D(num_channel=num_channel, num_filter=num_filter, num_multiplier=num_multiplier,
            window_size=window_size, stride_size=stride_size, padding_type=padding_type, activation=activation, dropout=dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            trainable=trainable, scope=scope)
    elif conv_type == "sep_2d":
        conv_layer = SeparableConv2D(num_channel=num_channel, num_filter=num_filter, num_multiplier=num_multiplier,
            window_size=window_size, stride_size=stride_size, padding_type=padding_type, activation=activation, dropout=dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            trainable=trainable, scope=scope)
    elif conv_type == "multi_sep_1d":
        conv_layer = MultiSeparableConv1D(num_channel=num_channel, num_filter=num_filter, num_multiplier=num_multiplier,
            window_size=window_size, stride_size=stride_size, padding_type=padding_type, activation=activation, dropout=dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            trainable=trainable, scope=scope)
    elif conv_type == "multi_sep_2d":
        conv_layer = MultiSeparableConv2D(num_channel=num_channel, num_filter=num_filter, num_multiplier=num_multiplier,
            window_size=window_size, stride_size=stride_size, padding_type=padding_type, activation=activation, dropout=dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported convolution type {0}".format(conv_type))
    
    return conv_layer

def create_pooling_layer(pooling_type,
                         num_gpus,
                         default_gpu_id):
    """create pooling layer"""
    scope = "pooling/{0}".format(pooling_type)
    if pooling_type == "max":
        pooling_layer = MaxPooling(num_gpus=num_gpus, default_gpu_id=default_gpu_id, scope=scope)
    elif pooling_type == "avg":
        pooling_layer = AveragePooling(num_gpus=num_gpus, default_gpu_id=default_gpu_id, scope=scope)
    else:
        raise ValueError("unsupported pooling type {0}".format(pooling_type))
    
    return pooling_layer

def create_dense_layer(num_layer,
                       unit_dim,
                       activation,
                       dropout,
                       num_gpus,
                       default_gpu_id,
                       trainable):
    """create dense layer"""
    if num_layer > 1:
        dense_layer = Dense(unit_dim=unit_dim, activation=activation, dropout=dropout,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope="dense")
    else:
        dense_layer = StackedDense(num_layer=num_layer, unit_dim=unit_dim, activation=activation, dropout=dropout,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope="stacked_dense")
    
    return dense_layer

def create_highway_layer(num_layer,
                         unit_dim,
                         activation,
                         dropout,
                         num_gpus,
                         default_gpu_id,
                         trainable):
    """create highway layer"""
    if num_layer > 1:
        highway_layer = Highway(unit_dim=unit_dim, activation=activation, dropout=dropout,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope="highway")
    else:
        highway_layer = StackedHighway(num_layer=num_layer, unit_dim=unit_dim, activation=activation, dropout=dropout,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope="stacked_highway")
    
    return highway_layer

def create_recurrent_layer(recurrent_type,
                           num_layer,
                           unit_dim,
                           cell_type,
                           activation,
                           dropout,
                           forget_bias,
                           residual_connect,
                           num_gpus,
                           default_gpu_id,
                           trainable):
    """create recurrent layer"""
    scope = "recurrent/{0}".format(recurrent_type)
    if recurrent_type == "uni":
        recurrent_layer = RNN(num_layer=num_layer, unit_dim=unit_dim, cell_type=cell_type,
            activation=activation, dropout=dropout, forget_bias=forget_bias, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif recurrent_type == "bi":
        recurrent_layer = BiRNN(num_layer=num_layer, unit_dim=unit_dim, cell_type=cell_type,
            activation=activation, dropout=dropout, forget_bias=forget_bias, residual_connect=residual_connect,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported recurrent type {0}".format(recurrent_type))
    
    return recurrent_layer

def create_attention_layer(attention_type,
                           src_dim,
                           trg_dim,
                           att_dim,
                           score_type,
                           is_self,
                           external_matrix,
                           num_gpus,
                           default_gpu_id,
                           trainable):
    """create attention layer"""
    scope = "attention/{0}".format(attention_type)
    if attention_type == "att":
        attention_layer = Attention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim, score_type=score_type, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif attention_type == "max_att":
        attention_layer = MaxAttention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim, score_type=score_type, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif attention_type == "head_att":
        attention_layer = HeadAttention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim, score_type=score_type, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    elif attention_type == "multi_head_att":
        attention_layer = MultiHeadAttention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim, score_type=score_type, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id, trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported attention type {0}".format(attention_type))
    
    return attention_layer
