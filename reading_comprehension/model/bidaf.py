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
            self.logger.log_print("# build graph for bidaf model")
            """build representation layer for bidaf model"""
            self.logger.log_print("# build question representation layer for bidaf model")
            (question_feat, question_feat_mask,
                question_word_embedding_placeholder) = self._build_representation_layer(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask)
            self.logger.log_print("# build context representation layer for bidaf model")
            (context_feat, context_feat_mask,
                context_word_embedding_placeholder) = self._build_representation_layer(context_word, context_word_mask,
                    context_subword, context_subword_mask, context_char, context_char_mask)
            self.question_feat = question_feat
            self.question_feat_mask = question_feat_mask
            self.question_word_embedding_placeholder = question_word_embedding_placeholder
            self.context_feat = context_feat
            self.context_feat_mask = context_feat_mask
            self.context_word_embedding_placeholder = context_word_embedding_placeholder
            
            """build understanding layer for bidaf model"""
            (question_understanding, context_understanding, question_understanding_mask,
                context_understanding_mask) = self._build_understanding_layer(self.question_feat,
                    self.context_feat, self.question_feat_mask, self.context_feat_mask)
            self.question_understanding = question_understanding
            self.question_understanding_mask = question_understanding_mask
            self.context_understanding = context_understanding
            self.context_understanding_mask = context_understanding_mask
            
            """build interaction layer for bidaf model"""
            answer_interaction, answer_interaction_mask = self._build_interaction_layer(self.question_understanding,
                self.context_understanding, self.question_understanding_mask, self.context_understanding_mask)
            self.answer_interaction = answer_interaction
            self.answer_interaction_mask = answer_interaction_mask
            
            """build modeling layer for bidaf model"""
            self.logger.log_print("# build answer modeling layer for bidaf model")
            answer_modeling, answer_modeling_mask = self._build_modeling_layer(self.answer_interaction, self.answer_interaction_mask)
            self.answer_modeling = answer_interaction
            self.answer_modeling_mask = answer_interaction_mask
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _build_fusion_result(self,
                             input_data_list,
                             input_mask_list,
                             input_unit_dim,
                             output_unit_dim,
                             fusion_type,
                             fusion_num_layer,
                             fusion_hidden_activation,
                             fusion_trainable):
        """build fusion layer for bidaf model"""
        input_data = tf.concat(input_data_list, axis=-1)
        output_fusion_mask = tf.reduce_max(tf.concat(input_mask_list, axis=-1), axis=-1, keep_dims=True)
        
        if input_unit_dim != output_unit_dim:
            convert_layer = tf.layers.Dense(units=output_unit_dim, activation=None, trainable=fusion_trainable)
            input_data = convert_layer(input_data)
        
        if fusion_type == "dense":
            activation = create_activation_function(fusion_hidden_activation)
            fusion_layer = tf.layers.Dense(units=output_unit_dim,
                activation=activation, trainable=fusion_trainable)
            output_fusion = fusion_layer(input_data)
        elif fusion_type == "highway":
            fusion_layer = create_highway_layer("fc", fusion_num_layer,
                output_unit_dim, fusion_hidden_activation, fusion_trainable)
            output_fusion = fusion_layer(input_data)
        else:
            output_fusion = input_data
        
        return output_fusion, output_fusion_mask
    
    def _build_word_feat(self,
                         input_word,
                         input_word_mask,
                         word_vocab_size,
                         word_embed_dim,
                         word_embed_pretrained,
                         word_feat_trainable):
        """build word-level featurization for bidaf model"""
        with tf.variable_scope("featurization/word", reuse=tf.AUTO_REUSE):
            word_embedding_layer = create_embedding_layer(word_vocab_size,
                word_embed_dim, word_embed_pretrained, word_feat_trainable)
            input_word_embedding, word_embedding_placeholder = word_embedding_layer(input_word)
            
            input_word_feat = tf.squeeze(input_word_embedding, axis=-2)
            input_word_feat_mask = tf.squeeze(input_word_mask, axis=-1)
            
            return input_word_feat, input_word_feat_mask, word_embedding_placeholder
    
    def _build_subword_feat(self,
                            input_subword,
                            input_subword_mask,
                            subword_vocab_size,
                            subword_embed_dim,
                            subword_feat_trainable,
                            subword_max_length,
                            subword_window_size,
                            subword_hidden_activation,
                            subword_pooling_type):
        """build subword-level featurization for bidaf model"""
        with tf.variable_scope("featurization/subword", reuse=tf.AUTO_REUSE):
            subword_embedding_layer = create_embedding_layer(subword_vocab_size,
                subword_embed_dim, False, subword_feat_trainable)
            input_subword_embedding, _ = subword_embedding_layer(input_subword)
            
            subword_conv_layer = create_convolution_layer("2d", subword_embed_dim, subword_embed_dim,
                subword_window_size, 1, "SAME", subword_hidden_activation, subword_feat_trainable)
            input_subword_conv = subword_conv_layer(input_subword_embedding)
            
            subword_pooling_layer = create_pooling_layer(subword_pooling_type)
            input_subword_pool, input_subword_pool_mask = subword_pooling_layer(input_subword_conv, input_subword_mask)
            input_subword_feat = input_subword_pool
            input_subword_feat_mask = input_subword_pool_mask
            
            return input_subword_feat, input_subword_feat_mask
    
    def _build_char_feat(self,
                         input_char,
                         input_char_mask,
                         char_vocab_size,
                         char_embed_dim,
                         char_feat_trainable,
                         char_max_length,
                         char_window_size,
                         char_hidden_activation,
                         char_pooling_type):
        """build char-level featurization for bidaf model"""
        with tf.variable_scope("featurization/char", reuse=tf.AUTO_REUSE):
            char_embedding_layer = create_embedding_layer(char_vocab_size,
                char_embed_dim, False, char_feat_trainable)
            input_char_embedding, _ = char_embedding_layer(input_char)
            
            char_conv_layer = create_convolution_layer("2d", char_embed_dim, char_embed_dim,
                char_window_size, 1, "SAME", char_hidden_activation, char_feat_trainable)
            input_char_conv = char_conv_layer(input_char_embedding)
            
            char_pooling_layer = create_pooling_layer(char_pooling_type)
            input_char_pool, input_char_pool_mask = char_pooling_layer(input_char_conv, input_char_mask)
            input_char_feat = input_char_pool
            input_char_feat_mask = input_char_pool_mask
            
            return input_char_feat, input_char_feat_mask
    
    def _build_representation_layer(self,
                                    input_word,
                                    input_word_mask,
                                    input_subword,
                                    input_subword_mask,
                                    input_char,
                                    input_char_mask):
        """build representation layer for bidaf model"""
        word_vocab_size = self.hyperparams.data_word_vocab_size
        word_embed_dim = self.hyperparams.model_representation_word_embed_dim
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        word_feat_trainable = self.hyperparams.model_representation_word_feat_trainable
        word_feat_enable = self.hyperparams.model_representation_word_feat_enable
        subword_vocab_size = self.hyperparams.data_subword_vocab_size
        subword_embed_dim = self.hyperparams.model_representation_subword_embed_dim
        subword_feat_trainable = self.hyperparams.model_representation_subword_feat_trainable
        subword_max_length = self.hyperparams.data_max_subword_length
        subword_window_size = self.hyperparams.model_representation_subword_window_size
        subword_hidden_activation = self.hyperparams.model_representation_subword_hidden_activation
        subword_pooling_type = self.hyperparams.model_representation_subword_pooling_type
        subword_feat_enable = self.hyperparams.model_representation_subword_feat_enable
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_feat_trainable = self.hyperparams.model_representation_char_feat_trainable
        char_max_length = self.hyperparams.data_max_char_length
        char_window_size = self.hyperparams.model_representation_char_window_size
        char_hidden_activation = self.hyperparams.model_representation_char_hidden_activation
        char_pooling_type = self.hyperparams.model_representation_char_pooling_type
        char_feat_enable = self.hyperparams.model_representation_char_feat_enable
        fusion_type = self.hyperparams.model_representation_fusion_type
        fusion_num_layer = self.hyperparams.model_representation_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_representation_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_representation_fusion_hidden_activation
        fusion_trainable = self.hyperparams.model_representation_fusion_trainable
        
        input_feat_list = []
        input_feat_mask_list = []
        if word_feat_enable == True:
            (input_word_feat, input_word_feat_mask,
                word_embedding_placeholder) = self._build_word_feat(input_word, input_word_mask,
                    word_vocab_size, word_embed_dim, word_embed_pretrained, word_feat_trainable)
            input_feat_list.append(input_word_feat)
            input_feat_mask_list.append(input_word_feat_mask)
        else:
            word_embed_dim = 0
            word_embedding_placeholder = None
        
        if subword_feat_enable == True:
            input_subword_feat, input_subword_feat_mask = self._build_subword_feat(input_subword, input_subword_mask,
                subword_vocab_size, subword_embed_dim, subword_feat_trainable, subword_max_length,
                subword_window_size, subword_hidden_activation, subword_pooling_type)
            input_feat_list.append(input_subword_feat)
            input_feat_mask_list.append(input_subword_feat_mask)
        else:
            subword_embed_dim = 0
        
        if char_feat_enable == True:
            input_char_feat, input_char_feat_mask = self._build_char_feat(input_char, input_char_mask,
                char_vocab_size, char_embed_dim, char_feat_trainable, char_max_length,
                char_window_size, char_hidden_activation, char_pooling_type)
            input_feat_list.append(input_char_feat)
            input_feat_mask_list.append(input_char_feat_mask)
        else:
            char_embed_dim = 0
        
        feat_embed_dim = word_embed_dim + subword_embed_dim + char_embed_dim
        
        input_feat, input_feat_mask = self._build_fusion_result(input_feat_list,
            input_feat_mask_list, feat_embed_dim, fusion_unit_dim, fusion_type,
            fusion_num_layer, fusion_hidden_activation, fusion_trainable)
        
        return input_feat, input_feat_mask, word_embedding_placeholder

    def _build_understanding_layer(self,
                                   question_feat,
                                   context_feat,
                                   question_feat_mask,
                                   context_feat_mask):
        """build understanding layer for bidaf model"""
        feat_unit_dim = self.hyperparams.model_representation_fusion_unit_dim
        question_understanding_num_layer = self.hyperparams.model_understanding_question_num_layer
        question_understanding_unit_dim = self.hyperparams.model_understanding_question_unit_dim
        question_understanding_cell_type = self.hyperparams.model_understanding_question_cell_type
        question_understanding_hidden_activation = self.hyperparams.model_understanding_question_hidden_activation
        question_understanding_dropout = self.hyperparams.model_understanding_question_dropout
        question_understanding_forget_bias = self.hyperparams.model_understanding_question_forget_bias
        question_understanding_residual_connect = self.hyperparams.model_understanding_question_residual_connect
        question_understanding_trainable = self.hyperparams.model_understanding_question_trainable
        context_understanding_num_layer = self.hyperparams.model_understanding_context_num_layer
        context_understanding_unit_dim = self.hyperparams.model_understanding_context_unit_dim
        context_understanding_cell_type = self.hyperparams.model_understanding_context_cell_type
        context_understanding_hidden_activation = self.hyperparams.model_understanding_context_hidden_activation
        context_understanding_dropout = self.hyperparams.model_understanding_context_dropout
        context_understanding_forget_bias = self.hyperparams.model_understanding_context_forget_bias
        context_understanding_residual_connect = self.hyperparams.model_understanding_context_residual_connect
        context_understanding_trainable = self.hyperparams.model_understanding_context_trainable
        
        self.logger.log_print("# build question understanding layer for bidaf model")
        question_feat, question_feat_mask = self._build_fusion_result([question_feat], [question_feat_mask],
            feat_unit_dim, question_understanding_unit_dim, "pass", 0, None, question_understanding_trainable)
        
        question_understanding_layer = create_recurrent_layer("bi", question_understanding_num_layer,
            question_understanding_unit_dim, question_understanding_cell_type, question_understanding_hidden_activation,
            question_understanding_dropout, question_understanding_forget_bias, question_understanding_residual_connect,
            self.num_gpus, self.default_gpu_id, question_understanding_trainable)
        
        question_understanding, _ = question_understanding_layer(question_feat, question_feat_mask)
        question_understanding_mask = question_feat_mask
        
        self.logger.log_print("# build context understanding layer for bidaf model")
        context_feat, context_feat_mask = self._build_fusion_result([context_feat], [context_feat_mask],
            feat_unit_dim, context_understanding_unit_dim, "pass", 0, None, context_understanding_trainable)
        
        context_understanding_layer = create_recurrent_layer("bi", context_understanding_num_layer,
            context_understanding_unit_dim, context_understanding_cell_type, context_understanding_hidden_activation,
            context_understanding_dropout, context_understanding_forget_bias, context_understanding_residual_connect,
            self.num_gpus, self.default_gpu_id, context_understanding_trainable)
        
        context_understanding, _ = context_understanding_layer(context_feat, context_feat_mask)
        context_understanding_mask = context_feat_mask
        
        return question_understanding, context_understanding, question_understanding_mask, context_understanding_mask
    
    def _build_interaction_layer(self,
                                 question_understanding,
                                 context_understanding,
                                 question_understanding_mask,
                                 context_understanding_mask):
        """build interaction layer for bidaf model"""
        question_understanding_unit_dim = self.hyperparams.model_understanding_question_unit_dim * 2
        context_understanding_unit_dim = self.hyperparams.model_understanding_context_unit_dim * 2
        quesiton2context_interaction_unit_dim = self.hyperparams.model_interaction_quesiton2context_unit_dim
        quesiton2context_interaction_score_type = self.hyperparams.model_interaction_quesiton2context_score_type
        quesiton2context_interaction_trainable = self.hyperparams.model_interaction_quesiton2context_trainable
        quesiton2context_interaction_enable = self.hyperparams.model_interaction_quesiton2context_enable
        context2quesiton_interaction_unit_dim = self.hyperparams.model_interaction_context2quesiton_unit_dim
        context2quesiton_interaction_score_type = self.hyperparams.model_interaction_context2quesiton_score_type
        context2quesiton_interaction_trainable = self.hyperparams.model_interaction_context2quesiton_trainable
        context2quesiton_interaction_enable = self.hyperparams.model_interaction_context2quesiton_enable
        fusion_type = self.hyperparams.model_interaction_fusion_type
        fusion_num_layer = self.hyperparams.model_interaction_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_interaction_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_interaction_fusion_hidden_activation
        fusion_trainable = self.hyperparams.model_representation_fusion_trainable
        
        answer_interaction_list = [context_understanding]
        answer_interaction_mask_list = [context_understanding_mask]
        if context2quesiton_interaction_enable == True:
            self.logger.log_print("# build context2question interaction layer for bidaf model")
            context2quesiton_attention_layer = create_attention_layer("default", context_understanding_unit_dim,
                question_understanding_unit_dim, context2quesiton_interaction_unit_dim,
                context2quesiton_interaction_score_type, context2quesiton_interaction_trainable)
            
            context2quesiton_interaction = context2quesiton_attention_layer(context_understanding,
                question_understanding, context_understanding_mask, question_understanding_mask)
            context2quesiton_interaction_mask = context_understanding_mask
            self.context2quesiton_interaction = context2quesiton_interaction
            answer_interaction_list.append(context2quesiton_interaction)
            answer_interaction_mask_list.append(context2quesiton_interaction_mask)
        else:
            context2quesiton_interaction_unit_dim = 0
        
        if quesiton2context_interaction_enable == True:
            self.logger.log_print("# build question2context interaction layer for bidaf model")
            quesiton2context_attention_layer = create_attention_layer("max_att", context_understanding_unit_dim,
                question_understanding_unit_dim, quesiton2context_interaction_unit_dim,
                quesiton2context_interaction_score_type, quesiton2context_interaction_trainable)
            
            quesiton2context_interaction = quesiton2context_attention_layer(context_understanding,
                question_understanding, context_understanding_mask, question_understanding_mask)
            quesiton2context_interaction_mask = context_understanding_mask
            self.quesiton2context_interaction = quesiton2context_interaction
            answer_interaction_list.append(quesiton2context_interaction)
            answer_interaction_mask_list.append(quesiton2context_interaction_mask)
        else:
            quesiton2context_interaction_unit_dim = 0
        
        answer_interaction_unit_dim = context2quesiton_interaction_unit_dim + quesiton2context_interaction_unit_dim
        
        answer_interaction, answer_interaction_mask = self._build_fusion_result(answer_interaction_list,
            answer_interaction_mask_list, answer_interaction_unit_dim, fusion_unit_dim, fusion_type,
            fusion_num_layer, fusion_hidden_activation, fusion_trainable)
        
        return answer_interaction, answer_interaction_mask
    
    def _build_modeling_layer(self,
                              answer_interaction,
                              answer_interaction_mask):
        """build modeling layer for bidaf model"""
        interaction_unit_dim = self.hyperparams.model_interaction_fusion_unit_dim
        answer_modeling_num_layer = self.hyperparams.model_modeling_answer_num_layer
        answer_modeling_unit_dim = self.hyperparams.model_modeling_answer_unit_dim
        answer_modeling_cell_type = self.hyperparams.model_modeling_answer_cell_type
        answer_modeling_hidden_activation = self.hyperparams.model_modeling_answer_hidden_activation
        answer_modeling_dropout = self.hyperparams.model_modeling_answer_dropout
        answer_modeling_forget_bias = self.hyperparams.model_modeling_answer_forget_bias
        answer_modeling_residual_connect = self.hyperparams.model_modeling_answer_residual_connect
        answer_modeling_trainable = self.hyperparams.model_modeling_answer_trainable
        
        answer_modeling, answer_modeling_mask = self._build_fusion_result([answer_interaction], [answer_interaction_mask],
            interaction_unit_dim, answer_modeling_unit_dim, "pass", 0, None, answer_modeling_trainable)
        
        answer_modeling_layer = create_recurrent_layer("bi", answer_modeling_num_layer,
            answer_modeling_unit_dim, answer_modeling_cell_type, answer_modeling_hidden_activation,
            answer_modeling_dropout, answer_modeling_forget_bias, answer_modeling_residual_connect,
            self.num_gpus, self.default_gpu_id, answer_modeling_trainable)
        
        answer_modeling, _ = answer_modeling_layer(answer_interaction, answer_interaction_mask)
        answer_modeling_mask = answer_interaction_mask
        
        return answer_modeling, answer_modeling_mask
    
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
