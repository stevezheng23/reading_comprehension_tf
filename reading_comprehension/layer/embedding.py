import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Embedding", "PretrainedEmbedding"]

class Embedding(object):
    """Embedding layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 trainable=True,
                 scope="embedding"):
        """initialize embedding layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            initializer = create_variable_initializer("glorot_uniform")
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embed_dim],
                initializer=initializer, trainable=self.trainable, dtype=tf.float32)
            self.embedding_placeholder = None
    
    def __call__(self,
                 input_data):
        """call embedding layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            output_embedding = tf.nn.embedding_lookup(self.embedding, input_data)
        
        return output_embedding
    
    def get_embedding_placeholder(self):
        """gert pretrained embedding placeholder"""
        return self.embedding_placeholder

class PretrainedEmbedding(object):
    """Pretrained Embedding layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 trainable=True,
                 scope="pretrained_embedding"):
        """initialize pretrained embedding layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            initializer = create_variable_initializer("zero")
            embedding = tf.get_variable("pretrained_embedding", shape=[self.vocab_size, self.embed_dim],
                initializer=initializer, trainable=self.trainable, dtype=tf.float32)
            self.embedding_placeholder = tf.placeholder(name="embedding_placeholder",
                shape=[self.vocab_size, self.embed_dim], dtype=tf.float32)
            self.embedding = embedding.assign(embedding_placeholder)
    
    def __call__(self,
                 input_data):
        """call pretrained embedding layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            output_embedding = tf.nn.embedding_lookup(self.embedding, input_data)
        
        return output_embedding
    
    def get_embedding_placeholder(self):
        """gert pretrained embedding placeholder"""
        return self.embedding_placeholder
