import collections
import os.path

import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *
from util.layer_util import *

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
            (input_question_feat, input_question_feat_mask, question_feat_embed_dim,
                word_embedding_placeholder) = self._build_input_feat(input_question_word,
                    input_question_word_mask, input_question_subword, input_question_subword_mask,
                    input_question_char, input_question_char_mask)
            self.input_question_feat = input_question_feat
            self.input_question_feat_mask = input_question_feat_mask
            self.question_feat_embed_dim = question_feat_embed_dim
            self.word_embedding_placeholder = word_embedding_placeholder
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _build_input_feat(self,
                          input_word,
                          input_word_mask,
                          input_subword,
                          input_subword_mask,
                          input_char,
                          input_char_mask):
        """build input featurization layer for reading comprehension base model"""
        word_feat_enable = self.hyperparams.model_representation_word_feat_enable
        subword_feat_enable = self.hyperparams.model_representation_subword_feat_enable
        char_feat_enable = self.hyperparams.model_representation_char_feat_enable
        fusion_type = self.hyperparams.model_representation_fusion_type
        fusion_trainable = self.hyperparams.model_representation_fusion_trainable
        fusion_num_layer = self.hyperparams.model_representation_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_representation_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_representation_fusion_hidden_activation
        
        input_feat_list = []
        input_feat_mask_list = []
        if word_feat_enable == True:
            (input_word_feat, input_word_feat_mask, word_embed_dim,
                word_embedding_placeholder) = self._build_word_feat(input_word, input_word_mask)
            input_feat_list.append(input_word_feat)
            input_feat_mask_list.append(input_word_feat_mask)
        else:
            input_word_feat = None
            input_word_mask = None
            word_embed_dim = 0
            word_embedding_placeholder = None
        
        if subword_feat_enable == True:
            (input_subword_feat, input_subword_feat_mask,
                subword_embed_dim) = self._build_subword_feat(input_subword, input_subword_mask)
            input_feat_list.append(input_subword_feat)
            input_feat_mask_list.append(input_subword_feat_mask)
        else:
            input_subword_feat = None
            input_subword_mask = None
            subword_embed_dim = 0
        
        if char_feat_enable == True:
            (input_char_feat, input_char_feat_mask,
                char_embed_dim) = self._build_char_feat(input_char, input_char_mask)
            input_feat_list.append(input_char_feat)
            input_feat_mask_list.append(input_char_feat_mask)
        else:
            input_char_feat = None
            input_char_mask = None
            char_embed_dim = 0
        
        input_feat = tf.concat(input_feat_list, axis=-1)
        input_feat_mask = tf.reduce_max(tf.concat(input_feat_mask_list, axis=-1), axis=-1)
        feat_embed_dim = word_embed_dim + subword_embed_dim + char_embed_dim
        
        if feat_embed_dim != fusion_unit_dim:
            convert_activation = create_activation_function(fusion_hidden_activation)
            convert_layer = tf.layers.Dense(units=fusion_unit_dim, activation=convert_activation, trainable=fusion_trainable)
            input_feat = convert_layer(input_feat)
        
        if fusion_type == "highway":
            highway_layer = create_highway_layer("full_connected", fusion_num_layer,
                fusion_unit_dim, fusion_hidden_activation, fusion_trainable)
            input_feat = highway_layer(input_feat)
        
        return input_feat, input_feat_mask, feat_embed_dim, word_embedding_placeholder
    
    def _build_word_feat(self,
                         input_word,
                         input_word_mask):
        """build word-level featurization layer for reading comprehension base model"""
        word_vocab_size = self.hyperparams.data_word_vocab_size
        word_embed_dim = self.hyperparams.model_representation_word_embed_dim
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        word_feat_trainable = self.hyperparams.model_representation_word_feat_trainable
        
        with tf.variable_scope("featurization/word", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create word-level featurization for reading comprehension base model")
            word_embedding_layer = create_embedding_layer(word_vocab_size,
                word_embed_dim, word_embed_pretrained, word_feat_trainable)
            input_word_embedding, word_embedding_placeholder = word_embedding_layer(input_word)
            
            input_word_feat = tf.squeeze(input_word_embedding, axis=-2)
            input_word_feat_mask = tf.squeeze(input_word_mask, axis=-1)
            
            return input_word_feat, input_word_feat_mask, word_embed_dim, word_embedding_placeholder
    
    def _build_subword_feat(self,
                            input_subword,
                            input_subword_mask):
        """build subword-level featurization layer for reading comprehension base model"""
        subword_vocab_size = self.hyperparams.data_subword_vocab_size
        subword_embed_dim = self.hyperparams.model_representation_subword_embed_dim
        subword_feat_trainable = self.hyperparams.model_representation_subword_feat_trainable
        subword_max_length = self.hyperparams.data_max_subword_length
        subword_window_size = self.hyperparams.model_representation_subword_window_size
        subword_hidden_activation = self.hyperparams.model_representation_subword_hidden_activation
        subword_pooling_type = self.hyperparams.model_representation_subword_pooling_type
        
        with tf.variable_scope("featurization/subword", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create subword-level featurization for reading comprehension base model")
            subword_embedding_layer = create_embedding_layer(subword_vocab_size,
                subword_embed_dim, False, subword_feat_trainable)
            input_subword_embedding, _ = subword_embedding_layer(input_subword)
            
            subword_conv_layer = create_convolution_layer("2d", subword_embed_dim, subword_embed_dim,
                subword_window_size, 1, "SAME", subword_hidden_activation, subword_feat_trainable)
            input_subword_conv = subword_conv_layer(input_subword_embedding)
            
            subword_pooling_layer = create_pooling_layer(subword_pooling_type)
            input_subword_feat, input_subword_feat_mask = subword_pooling_layer(input_subword_conv, input_subword_mask)
        
        return input_subword_feat, input_subword_feat_mask, subword_embed_dim
    
    def _build_char_feat(self,
                         input_char,
                         input_char_mask):
        """build char-level featurization layer for reading comprehension base model"""
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_feat_trainable = self.hyperparams.model_representation_char_feat_trainable
        char_max_length = self.hyperparams.data_max_char_length
        char_window_size = self.hyperparams.model_representation_char_window_size
        char_hidden_activation = self.hyperparams.model_representation_char_hidden_activation
        char_pooling_type = self.hyperparams.model_representation_char_pooling_type
        
        with tf.variable_scope("featurization/char", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create char-level featurization for reading comprehension base model")
            char_embedding_layer = create_embedding_layer(char_vocab_size,
                char_embed_dim, False, char_feat_trainable)
            input_char_embedding, _ = char_embedding_layer(input_char)
            
            char_conv_layer = create_convolution_layer("2d", char_embed_dim, char_embed_dim,
                char_window_size, 1, "SAME", char_hidden_activation, char_feat_trainable)
            input_char_conv = char_conv_layer(input_char_embedding)
            
            char_pooling_layer = create_pooling_layer(char_pooling_type)
            input_char_feat, input_char_feat_mask = char_pooling_layer(input_char_conv, input_char_mask)
        
        return input_char_feat, input_char_feat_mask, char_embed_dim
    
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
