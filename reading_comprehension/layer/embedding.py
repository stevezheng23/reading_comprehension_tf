import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["Embedding"]

class Embedding(object):
    """Embedding layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 trainable=True,
                 pretrained=False,
                 scope="embedding"):
        """initialize embedding layer"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.trainable = trainable
            self.pretrained = pretrained
            
            if self.pretrained == True:
                self.embedding, self.embedding_placeholder = create_pretrained_embedding(self.vocab_size,
                    self.embed_dim, self.trainable)
            else:
                self.embedding = create_embedding(self.vocab_size, self.embed_dim, self.trainable)
                self.embedding_placeholder = None
    
    def __call__(self,
                 input_data):
        """generate embedding layer output"""
        input_embedding = tf.nn.embedding_lookup(self.embedding, input_data)
        
        return input_embedding, self.embedding_placeholder
