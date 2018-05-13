import collections
import os.path

import numpy as np
import tensorflow as tf

from util.reading_comprehension_util import *
from util.layer_util import *

from model.base_model import *

__all__ = ["BiDAF"]

class BiDAF(BaseModel):
    """bi-directional attention flow model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="bidaf"):
        """initialize bidaf model"""        
        super(BiDAF, self).__init__(logger=logger, hyperparams=hyperparams,
            data_pipeline=data_pipeline, mode=mode, scope=scope)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer, trainable=False)
            
            """get batch input from data pipeline"""
            question_word = self.data_pipeline.input_question_word
            question_subword = self.data_pipeline.input_question_subword
            question_char = self.data_pipeline.input_question_char
            question_word_mask = self.data_pipeline.input_question_word_mask
            question_subword_mask = self.data_pipeline.input_question_subword_mask
            question_char_mask = self.data_pipeline.input_question_char_mask
            
            context_word = self.data_pipeline.input_context_word
            context_subword = self.data_pipeline.input_context_subword
            context_char = self.data_pipeline.input_context_char
            context_word_mask = self.data_pipeline.input_context_word_mask
            context_subword_mask = self.data_pipeline.input_context_subword_mask
            context_char_mask = self.data_pipeline.input_context_char_mask
            
            """build graph for bidaf model"""
            """build representation layer for bidaf model"""
            (question_feat, question_feat_mask, question_feat_embed_dim,
                question_word_embedding_placeholder) = self._build_representation_layer(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask)
            self.question_feat = question_feat
            self.question_feat_mask = question_feat_mask
            self.question_feat_embed_dim = question_feat_embed_dim
            self.question_word_embedding_placeholder = question_word_embedding_placeholder
            
            (context_feat, context_feat_mask, context_feat_embed_dim,
                context_word_embedding_placeholder) = self._build_representation_layer(context_word, context_word_mask,
                    context_subword, context_subword_mask, context_char, context_char_mask)
            self.context_feat = context_feat
            self.context_feat_mask = context_feat_mask
            self.context_feat_embed_dim = context_feat_embed_dim
            self.context_word_embedding_placeholder = context_word_embedding_placeholder
            
            """build understanding layer for bidaf model"""
            (question_understanding_output,
                question_understanding_final_state) = self._build_question_understanding_layer(self.question_feat,
                    self.question_feat_mask)
            self.question_understanding_output = question_understanding_output
            self.question_understanding_final_state = question_understanding_final_state
            
            (context_understanding_output,
                context_understanding_final_state) = self._build_context_understanding_layer(self.context_feat,
                    self.context_feat_mask)
            self.context_understanding_output = context_understanding_output
            self.context_understanding_final_state = context_understanding_final_state
            
            """build interaction layer for bidaf model"""
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _build_representation_layer(self,
                                    input_word,
                                    input_word_mask,
                                    input_subword,
                                    input_subword_mask,
                                    input_char,
                                    input_char_mask):
        """build representation layer for bidaf model"""
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
            highway_layer = create_highway_layer("fc", fusion_num_layer,
                fusion_unit_dim, fusion_hidden_activation, fusion_trainable)
            input_feat = highway_layer(input_feat)
        
        return input_feat, input_feat_mask, feat_embed_dim, word_embedding_placeholder
    
    def _build_word_feat(self,
                         input_word,
                         input_word_mask):
        """build word-level featurization layer for bidaf model"""
        word_vocab_size = self.hyperparams.data_word_vocab_size
        word_embed_dim = self.hyperparams.model_representation_word_embed_dim
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        word_feat_trainable = self.hyperparams.model_representation_word_feat_trainable
        
        with tf.variable_scope("featurization/word", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create word-level featurization for bidaf model")
            word_embedding_layer = create_embedding_layer(word_vocab_size,
                word_embed_dim, word_embed_pretrained, word_feat_trainable)
            input_word_embedding, word_embedding_placeholder = word_embedding_layer(input_word)
            
            input_word_feat = tf.squeeze(input_word_embedding, axis=-2)
            input_word_feat_mask = input_word_mask
            
            return input_word_feat, input_word_feat_mask, word_embed_dim, word_embedding_placeholder
    
    def _build_subword_feat(self,
                            input_subword,
                            input_subword_mask):
        """build subword-level featurization layer for bidaf model"""
        subword_vocab_size = self.hyperparams.data_subword_vocab_size
        subword_embed_dim = self.hyperparams.model_representation_subword_embed_dim
        subword_feat_trainable = self.hyperparams.model_representation_subword_feat_trainable
        subword_max_length = self.hyperparams.data_max_subword_length
        subword_window_size = self.hyperparams.model_representation_subword_window_size
        subword_hidden_activation = self.hyperparams.model_representation_subword_hidden_activation
        subword_pooling_type = self.hyperparams.model_representation_subword_pooling_type
        
        with tf.variable_scope("featurization/subword", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create subword-level featurization for bidaf model")
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
        """build char-level featurization layer for bidaf model"""
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_feat_trainable = self.hyperparams.model_representation_char_feat_trainable
        char_max_length = self.hyperparams.data_max_char_length
        char_window_size = self.hyperparams.model_representation_char_window_size
        char_hidden_activation = self.hyperparams.model_representation_char_hidden_activation
        char_pooling_type = self.hyperparams.model_representation_char_pooling_type
        
        with tf.variable_scope("featurization/char", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create char-level featurization for bidaf model")
            char_embedding_layer = create_embedding_layer(char_vocab_size,
                char_embed_dim, False, char_feat_trainable)
            input_char_embedding, _ = char_embedding_layer(input_char)
            
            char_conv_layer = create_convolution_layer("2d", char_embed_dim, char_embed_dim,
                char_window_size, 1, "SAME", char_hidden_activation, char_feat_trainable)
            input_char_conv = char_conv_layer(input_char_embedding)
            
            char_pooling_layer = create_pooling_layer(char_pooling_type)
            input_char_feat, input_char_feat_mask = char_pooling_layer(input_char_conv, input_char_mask)
        
        return input_char_feat, input_char_feat_mask, char_embed_dim
    
    def _build_question_understanding_layer(self,
                                            question_feat,
                                            question_feat_mask):
        """build question understanding layer for bidaf model"""
        question_understanding_num_layer = self.hyperparams.model_question_understanding_num_layer
        question_understanding_unit_dim = self.hyperparams.model_question_understanding_unit_dim
        question_understanding_cell_type = self.hyperparams.model_question_understanding_cell_type
        question_understanding_hidden_activation = self.hyperparams.model_question_understanding_hidden_activation
        question_understanding_dropout = self.hyperparams.model_question_understanding_dropout
        question_understanding_forget_bias = self.hyperparams.model_question_understanding_forget_bias
        question_understanding_residual_connect = self.hyperparams.model_question_understanding_residual_connect
        question_understanding_trainable = self.hyperparams.model_question_understanding_trainable
        
        question_understanding_layer = create_recurrent_layer("bi", question_understanding_num_layer,
            question_understanding_unit_dim, question_understanding_cell_type, question_understanding_hidden_activation,
            question_understanding_dropout, question_understanding_forget_bias, question_understanding_residual_connect,
            self.num_gpus, self.default_gpu_id, question_understanding_trainable)
        
        question_output, question_final_state = question_understanding_layer(question_feat, question_feat_mask)
        
        return question_output, question_final_state
    
    def _build_context_understanding_layer(self,
                                           context_feat,
                                           context_feat_mask):
        """build context understanding layer for mrc base model"""
        context_understanding_num_layer = self.hyperparams.model_context_understanding_num_layer
        context_understanding_unit_dim = self.hyperparams.model_context_understanding_unit_dim
        context_understanding_cell_type = self.hyperparams.model_context_understanding_cell_type
        context_understanding_hidden_activation = self.hyperparams.model_context_understanding_hidden_activation
        context_understanding_dropout = self.hyperparams.model_context_understanding_dropout
        context_understanding_forget_bias = self.hyperparams.model_context_understanding_forget_bias
        context_understanding_residual_connect = self.hyperparams.model_context_understanding_residual_connect
        context_understanding_trainable = self.hyperparams.model_context_understanding_trainable
        
        context_understanding_layer = create_recurrent_layer("bi", context_understanding_num_layer,
            context_understanding_unit_dim, context_understanding_cell_type, context_understanding_hidden_activation,
            context_understanding_dropout, context_understanding_forget_bias, context_understanding_residual_connect,
            self.num_gpus, self.default_gpu_id, context_understanding_trainable)
        
        context_output, context_final_state = context_understanding_layer(context_feat, context_feat_mask)
        
        return context_output, context_final_state
    
    def _build_context2question_interaction_layer(self,
                                                  question_understanding,
                                                  context_understanding):
        """build context-to-question interaction layer for bidaf model"""
        pass
    
    def save(self,
             sess,
             global_step):
        """save checkpoint for bidaf model"""
        self.ckpt_saver.save(sess, self.ckpt_name, global_step=global_step)
    
    def restore(self,
                sess):
        """restore bidaf model from checkpoint"""
        ckpt_file = tf.train.latest_checkpoint(self.ckpt_dir)
        if ckpt_file is not None:
            self.ckpt_saver.restore(sess, ckpt_file)
        else:
            raise FileNotFoundError("latest checkpoint file doesn't exist")
