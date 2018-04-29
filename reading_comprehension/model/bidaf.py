import collections
import os.path

import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *

__all__ = ["TrainResult", "EvaluateResult", "InferResult", "BiDAF"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class EvaluateResult(collections.namedtuple("EvaluateResult", ("loss", "batch_size", "word_count"))):
    pass

class InferResult(collections.namedtuple("InferResult",
    ("logits", "sample_id", "sample_word", "sample_sentence", "batch_size", "summary"))):
    pass

class BiDAF(object):
    """bi-directional attention flow model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="bidaf"):
        """initialize bi-directional attention flow model"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.logger = logger
            self.hyperparams = hyperparams
            
            self.data_pipeline = data_pipeline
            self.mode = mode
            self.scope = scope
            
            self.num_gpus = self.hyperparams.device_num_gpus
            self.default_gpu_id = self.hyperparams.device_default_gpu_id
            self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
                .format(self.num_gpus, self.default_gpu_id))
            
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer, trainable=False)
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _build_word_embedding(self,
                              input_word):
        """build word embedding layer for bi-directional attention flow model"""
        word_vocab_size = self.hyperparams.data_word_vocab_size
        word_embed_dim = self.hyperparams.model_representation_word_embed_dim
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        
        with tf.variable_scope("word/embedding", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create word embedding for bi-directional attention flow model")
            word_embedding, word_embedding_placeholder = create_embedding(word_vocab_size,
                word_embed_dim, word_embed_pretrained)
            input_word_embedding = tf.nn.embedding_lookup(word_embedding, input_word)
            
            return input_word_embedding, word_embedding_placeholder
    
    def _build_char_embedding(self,
                              input_char):
        """build char embedding layer for bi-directional attention flow model"""
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_window_size = self.hyperparams.model_representation_char_window_size
        char_pooling_type = self.hyperparams.model_representation_char_pooling_type
        
        with tf.variable_scope("char/embedding", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create char embedding for bi-directional attention flow model")
            char_embedding, _ = create_embedding(char_vocab_size,
                char_embed_dim, False)
            input_char_embedding = tf.nn.embedding_lookup(char_embedding, input_char)
        
        with tf.variable_scope("char/conv", reuse=tf.AUTO_REUSE):
            (batch_size, max_word_length, max_char_length,
                 char_embed_dim) = tf.shape(input_char_embedding)
            input_char_embedding = tf.reshape(input_char_embedding,
                shape=[batch_size * max_word_length, max_char_length, char_embed_dim])
            filters = tf.get_variable("filter",
                shape=[char_window_size, char_embed_dim, char_embed_dim], dtype=tf.float32)
            input_char_embedding = tf.nn.conv1d(input_char_embedding,
                filters=filters, stride=1, padding="SAME", use_cudnn_on_gpu=False)
            input_char_embedding = tf.reshape(input_char_embedding,
                shape=[batch_size, max_word_length, max_char_length, char_embed_dim])
        
        with tf.variable_scope("char/pool", reuse=tf.AUTO_REUSE):
            (batch_size, max_word_length, max_char_length,
                 char_embed_dim) = tf.shape(input_char_embedding)
            input_char_embedding = tf.nn.pool(input_char_embedding,
                window_shape=[1, 1, 1, char_embed_dim], pooling_type=char_pooling_type, padding="VALID")
        
        return input_char_embedding
    
    def save(self,
             sess,
             global_step):
        """save checkpoint for bi-directional attention flow model"""
        self.ckpt_saver.save(sess, self.ckpt_name, global_step=global_step)
    
    def restore(self,
                sess):
        """restore bi-directional attention flow model from checkpoint"""
        ckpt_file = tf.train.latest_checkpoint(self.ckpt_dir)
        if ckpt_file is not None:
            self.ckpt_saver.restore(sess, ckpt_file)
        else:
            raise FileNotFoundError("latest checkpoint file doesn't exist")
