import numpy as np
import tensorflow as tf

__all__ = ["create_embedding", "create_pretrained_embedding",
           "create_variable_initializer", "create_activation_function"]

def create_embedding(vocab_size,
                     embedding_dim,
                     trainable=True,
                     data_type=tf.float32):
    """create embedding with initializer"""
    initializer = create_variable_initializer("glorot_uniform")
    embedding = tf.get_variable("embedding", shape=[vocab_size, embedding_dim],
        initializer=initializer, trainable=trainable, dtype=data_type)
    
    return embedding

def create_pretrained_embedding(vocab_size,
                                embedding_dim,
                                trainable=True,
                                data_type=tf.float32):
    """create embedding with pre-trained embedding"""
    initializer = create_variable_initializer("zero")
    embedding = tf.get_variable("pretrained_embedding", shape=[vocab_size, embedding_dim],
        initializer=initializer, trainable=trainable, dtype=data_type)
    embedding_placeholder = tf.placeholder(name="embedding_placeholder",
        shape=[vocab_size, embedding_dim], dtype=data_type)
    embedding = embedding.assign(embedding_placeholder)
    
    return embedding, embedding_placeholder

def create_variable_initializer(initializer_type,
                                random_seed=None,
                                data_type=tf.float32):
    if initializer_type == "zero":
        initializer = tf.zeros_initializer
    elif initializer_type == "orthogonal":
        initializer = tf.orthogonal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "random_uniform":
        initializer = tf.random_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_uniform":
        initializer = tf.glorot_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "random_normal":
        initializer = tf.random_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "truncated_normal":
        initializer = tf.truncated_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_normal":
        initializer = tf.glorot_normal_initializer(seed=random_seed, dtype=data_type)
    else:
        initializer = None
    
    return initializer

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
