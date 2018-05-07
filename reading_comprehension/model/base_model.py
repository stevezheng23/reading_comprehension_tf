import collections
import os.path

import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *
from layer.embedding import *
from layer.convolution import *
from layer.pooling import *

__all__ = ["TrainResult", "EvaluateResult", "InferResult", "BaseModel"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class EvaluateResult(collections.namedtuple("EvaluateResult", ("loss", "batch_size", "word_count"))):
    pass

class InferResult(collections.namedtuple("InferResult",
    ("logits", "sample_id", "sample_word", "sample_sentence", "batch_size", "summary"))):
    pass

class BaseModel(object):
    """reading comprehension base model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="base"):
        """initialize reading comprehension base model"""
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
            
            input_question_word = self.data_pipeline.input_question_word
            input_question_subword = self.data_pipeline.input_question_subword
            input_question_char = self.data_pipeline.input_question_char
            input_question_word_mask = self.data_pipeline.input_question_word_mask
            input_question_subword_mask = self.data_pipeline.input_question_subword_mask
            input_question_char_mask = self.data_pipeline.input_question_char_mask
            (input_question_word_feat, input_question_subword_feat,
                input_question_char_feat, _) = self._build_input_feat(input_question_word,
                    input_question_subword, input_question_char, input_question_word_mask,
                    input_question_subword_mask, input_question_char_mask)
            self.input_question_word_feat = input_question_word_feat
            self.input_question_subword_feat = input_question_subword_feat
            self.input_question_char_feat = input_question_char_feat
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _build_input_feat(self,
                          input_word,
                          input_subword,
                          input_char,
                          input_word_mask,
                          input_subword_mask,
                          input_char_mask):
        """build input featurization layer for reading comprehension base model"""
        input_word_feat, word_embedding_placeholder = self._build_word_feat(input_word)
        input_subword_feat = None
        input_char_feat = self._build_char_feat(input_char, input_char_mask)
        
        return input_word_feat, input_subword_feat, input_char_feat, word_embedding_placeholder
    
    def _build_word_feat(self,
                         input_word):
        """build word-level featurization layer for reading comprehension base model"""
        word_vocab_size = self.hyperparams.data_word_vocab_size
        word_embed_dim = self.hyperparams.model_representation_word_embed_dim
        word_embed_trainable = self.hyperparams.model_representation_word_embed_trainable
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        
        with tf.variable_scope("featurization/word", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create word-level featurization for reading comprehension base model")
            word_embedding_layer = Embedding(vocab_size=word_vocab_size,
                embed_dim=word_embed_dim, trainable=word_embed_trainable, pretrained=word_embed_pretrained)
            input_word_embedding, word_embedding_placeholder = word_embedding_layer(input_word)
            
            input_word_feat = input_word_embedding
            
            return input_word_feat, word_embedding_placeholder
    
    def _build_char_feat(self,
                         input_char,
                         input_char_mask):
        """build char-level featurization layer for reading comprehension base model"""
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_embed_trainable = self.hyperparams.model_representation_char_embed_trainable
        char_max_length = self.hyperparams.data_max_char_length
        char_window_size = self.hyperparams.model_representation_char_window_size
        char_pooling_type = self.hyperparams.model_representation_char_pooling_type
        
        with tf.variable_scope("featurization/char", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create char-level featurization for reading comprehension base model")
            char_embedding_layer = Embedding(vocab_size=char_vocab_size,
                embed_dim=char_embed_dim, trainable=char_embed_trainable, pretrained=False)
            input_char_embedding, _ = char_embedding_layer(input_char)
            
            char_conv_layer = Conv2D(num_channel=char_embed_dim, num_filter=char_embed_dim,
                 window_size=char_window_size, stride_size=1, padding_type="SAME", trainable=True)
            self.input_char_embedding_shape = tf.shape(input_char_embedding)
            input_char_conv = char_conv_layer(input_char_embedding)
            
            char_pooling_layer = Pooling(pooling_type=char_pooling_type)
            input_char_pooling = char_pooling_layer(input_char_conv, input_char_mask)
            
            input_char_feat = input_char_pooling
        
        return input_char_feat
    
    def save(self,
             sess,
             global_step):
        """save checkpoint for reading comprehension base model"""
        self.ckpt_saver.save(sess, self.ckpt_name, global_step=global_step)
    
    def restore(self,
                sess):
        """restore reading comprehension base model from checkpoint"""
        ckpt_file = tf.train.latest_checkpoint(self.ckpt_dir)
        if ckpt_file is not None:
            self.ckpt_saver.restore(sess, ckpt_file)
        else:
            raise FileNotFoundError("latest checkpoint file doesn't exist")
