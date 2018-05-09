import numpy as np
import tensorflow as tf

from layer.embedding import *
from layer.convolution import *
from layer.pooling import *
from layer.highway import *

__all__ = ["create_embedding_layer", "create_convolution_layer", "create_pooling_layer",
           "create_highway_layer"]

def create_embedding_layer(vocab_size,
                           embed_dim,
                           pretrained,
                           trainable):
    """create pooling layer"""
    if pretrained == True:
        embed_layer = PretrainedEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, trainable=trainable)
    else:
        embed_layer = Embedding(vocab_size=vocab_size, embed_dim=embed_dim, trainable=trainable)
    
    return embed_layer

def create_convolution_layer(conv_type,
                             num_channel,
                             num_filter,
                             window_size,
                             stride_size,
                             padding_type,
                             activation,
                             trainable):
    """create convolution layer"""
    scope = "conv/{0}".format(conv_type)
    if conv_type == "1d":
        conv_layer = Conv1D(num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, trainable=trainable, scope=conv_type)
    elif conv_type == "2d":
        conv_layer = Conv2D(num_channel=num_channel, num_filter=num_filter, window_size=window_size, stride_size=stride_size,
            padding_type=padding_type, activation=activation, trainable=trainable, scope=conv_type)
    else:
        raise ValueError("unsupported convolution type {0}".format(conv_type))
    
    return conv_layer

def create_pooling_layer(pooling_type):
    """create pooling layer"""
    scope = "pooling/{0}".format(pooling_type)
    if pooling_type == "max":
        pooling_layer = MaxPooling(scope=scope)
    elif pooling_type == "avg":
        pooling_layer = AveragePooling(scope=scope)
    else:
        raise ValueError("unsupported pooling type {0}".format(pooling_type))
    
    return pooling_layer

def create_highway_layer(highway_type,
                         num_layer,
                         unit_dim,
                         activation,
                         trainable):
    """create highway layer"""
    if num_layer > 1:
        highway_layer = create_stacked_highway_layer(highway_type,
            num_layer, unit_dim, activation, trainable)
    else:
        highway_layer = create_single_highway_layer(highway_type,
            unit_dim, activation, trainable)
    
    return highway_layer

def create_single_highway_layer(highway_type,
                                unit_dim,
                                activation,
                                trainable):
    """create single highway layer"""
    scope = "highway/{0}".format(highway_type)
    if highway_type == "full_connected":
        highway_layer = Highway(unit_dim=unit_dim, activation=activation, trainable=trainable)
    else:
        raise ValueError("unsupported highway type {0}".format(highway_type))
    
    return highway_layer

def create_stacked_highway_layer(highway_type,
                                 num_layer,
                                 unit_dim,
                                 activation,
                                 trainable):
    """create stacked highway layer"""
    scope = "stacked_highway/{0}".format(highway_type)
    if highway_type == "full_connected":
        highway_layer = StackedHighway(num_layer=num_layer,
            unit_dim=unit_dim, activation=activation, trainable=trainable)
    else:
        raise ValueError("unsupported highway type {0}".format(highway_type))
    
    return highway_layer
