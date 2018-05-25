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
            answer_result = self.data_pipeline.input_answer
            answer_result_mask = self.data_pipeline.input_answer_mask
            
            """build graph for bidaf model"""
            self.logger.log_print("# build graph for bidaf model")
            (answer_start_output, answer_end_output, answer_start_output_mask,
                answer_end_output_mask, question_word_embedding_placeholder,
                context_word_embedding_placeholder) = self._build_graph(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask,
                    context_word, context_word_mask, context_subword, context_subword_mask, context_char, context_char_mask)
            self.answer_start_output = answer_start_output
            self.answer_end_output = answer_end_output
            self.answer_start_output_mask = answer_start_output_mask
            self.answer_end_output_mask = answer_end_output_mask
            self.question_word_embedding_placeholder = question_word_embedding_placeholder
            self.context_word_embedding_placeholder = context_word_embedding_placeholder
            
            if self.mode == "infer":
                """get infer answer"""
                self.infer_answer_start = tf.nn.softmax(tf.squeeze(
                    self.answer_start_output * self.answer_start_output_mask), dim=-1)
                self.infer_answer_end = tf.nn.softmax(tf.squeeze(
                    self.answer_end_output * self.answer_end_output_mask), dim=-1)
                self.infer_answer_start_mask = tf.squeeze(self.answer_start_output_mask)
                self.infer_answer_end_mask = tf.squeeze(self.answer_end_output_mask)
                
                """create infer summary"""
                self.infer_summary = self._get_infer_summary()
            
            if self.mode == "train":
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                answer_start_result = answer_result[:,0,:]
                answer_end_result = answer_result[:,1,:]
                start_loss = self._compute_loss(answer_start_result,
                    self.answer_start_output, self.answer_start_output_mask)
                end_loss = self._compute_loss(answer_end_result,
                    self.answer_end_output, self.answer_end_output_mask)
                self.train_loss = start_loss + end_loss
                
                """apply learning rate decay"""
                self.logger.log_print("# setup learning rate decay mechanism")
                self.learning_rate = tf.constant(self.hyperparams.train_optimizer_learning_rate)
                self.decayed_learning_rate = self._apply_learning_rate_decay(self.learning_rate)
                
                """initialize optimizer"""
                self.logger.log_print("# initialize optimizer")
                self.optimizer = self._initialize_optimizer(self.decayed_learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.update_model, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                """create train summary"""
                self.train_summary = self._get_train_summary()
            
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
        with tf.variable_scope("fusion", reuse=tf.AUTO_REUSE):
            input_data = tf.concat(input_data_list, axis=-1)
            output_fusion_mask = tf.reduce_max(tf.concat(input_mask_list, axis=-1), axis=-1, keep_dims=True)
            
            if input_unit_dim != output_unit_dim:
                convert_layer = tf.layers.Dense(units=output_unit_dim, activation=None, trainable=fusion_trainable)
                input_data = convert_layer(input_data)
            
            if fusion_type == "highway":
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
        with tf.variable_scope("feat/word", reuse=tf.AUTO_REUSE):
            word_embedding_layer = create_embedding_layer(word_vocab_size,
                word_embed_dim, word_embed_pretrained, word_feat_trainable)
            input_word_embedding, word_embedding_placeholder = word_embedding_layer(input_word)
            
            input_word_feat = tf.squeeze(input_word_embedding, axis=-2)
            input_word_feat_mask = input_word_mask
        
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
        with tf.variable_scope("feat/subword", reuse=tf.AUTO_REUSE):
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
        with tf.variable_scope("feat/char", reuse=tf.AUTO_REUSE):
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
        
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
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
        
        with tf.variable_scope("understanding/question", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build question understanding layer for bidaf model")
            question_understanding, question_understanding_mask = self._build_fusion_result([question_feat], [question_feat_mask],
                feat_unit_dim, question_understanding_unit_dim, "pass", 0, None, question_understanding_trainable)
            
            question_understanding_layer = create_recurrent_layer("bi", question_understanding_num_layer,
                question_understanding_unit_dim, question_understanding_cell_type, question_understanding_hidden_activation,
                question_understanding_dropout, question_understanding_forget_bias, question_understanding_residual_connect,
                self.num_gpus, self.default_gpu_id, question_understanding_trainable)
            
            question_understanding, _ = question_understanding_layer(question_understanding, question_understanding_mask)
        
        with tf.variable_scope("understanding/context", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build context understanding layer for bidaf model")
            context_understanding, context_understanding_mask = self._build_fusion_result([context_feat], [context_feat_mask],
                feat_unit_dim, context_understanding_unit_dim, "pass", 0, None, context_understanding_trainable)
            
            context_understanding_layer = create_recurrent_layer("bi", context_understanding_num_layer,
                context_understanding_unit_dim, context_understanding_cell_type, context_understanding_hidden_activation,
                context_understanding_dropout, context_understanding_forget_bias, context_understanding_residual_connect,
                self.num_gpus, self.default_gpu_id, context_understanding_trainable)
            
            context_understanding, _ = context_understanding_layer(context_understanding, context_understanding_mask)
        
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
        
        with tf.variable_scope("interaction", reuse=tf.AUTO_REUSE):
            answer_interaction_list = [context_understanding]
            answer_interaction_mask_list = [context_understanding_mask]
            
            with tf.variable_scope("context2question", reuse=tf.AUTO_REUSE):
                if context2quesiton_interaction_enable == True:
                    self.logger.log_print("# build context2question interaction layer for bidaf model")
                    context2quesiton_attention_layer = create_attention_layer("default", context_understanding_unit_dim,
                        question_understanding_unit_dim, context2quesiton_interaction_unit_dim,
                        context2quesiton_interaction_score_type, context2quesiton_interaction_trainable)
                    
                    context2quesiton_interaction = context2quesiton_attention_layer(context_understanding,
                        question_understanding, context_understanding_mask, question_understanding_mask)
                    context2quesiton_interaction_mask = context_understanding_mask
                    
                    answer_interaction_list.append(context2quesiton_interaction)
                    answer_interaction_mask_list.append(context2quesiton_interaction_mask)
                else:
                    question_understanding_unit_dim = 0
            
            with tf.variable_scope("question2context", reuse=tf.AUTO_REUSE):
                if quesiton2context_interaction_enable == True:
                    self.logger.log_print("# build question2context interaction layer for bidaf model")
                    quesiton2context_attention_layer = create_attention_layer("max_att", context_understanding_unit_dim,
                        question_understanding_unit_dim, quesiton2context_interaction_unit_dim,
                        quesiton2context_interaction_score_type, quesiton2context_interaction_trainable)
                    
                    quesiton2context_interaction = quesiton2context_attention_layer(context_understanding,
                        question_understanding, context_understanding_mask, question_understanding_mask)
                    quesiton2context_interaction_mask = context_understanding_mask
                    
                    answer_interaction_list.append(quesiton2context_interaction)
                    answer_interaction_mask_list.append(quesiton2context_interaction_mask)
                else:
                    context_understanding_unit_dim = 0
            
            answer_interaction_unit_dim = question_understanding_unit_dim + context_understanding_unit_dim * 2
            
            answer_interaction, answer_interaction_mask = self._build_fusion_result(answer_interaction_list,
                answer_interaction_mask_list, answer_interaction_unit_dim, fusion_unit_dim, fusion_type,
                fusion_num_layer, fusion_hidden_activation, fusion_trainable)
        
        return answer_interaction, answer_interaction_mask
    
    def _build_modeling_layer(self,
                              answer_interaction,
                              answer_interaction_mask):
        """build modeling layer for bidaf model"""
        answer_interaction_unit_dim = self.hyperparams.model_interaction_fusion_unit_dim
        answer_modeling_num_layer = self.hyperparams.model_modeling_answer_num_layer
        answer_modeling_unit_dim = self.hyperparams.model_modeling_answer_unit_dim
        answer_modeling_cell_type = self.hyperparams.model_modeling_answer_cell_type
        answer_modeling_hidden_activation = self.hyperparams.model_modeling_answer_hidden_activation
        answer_modeling_dropout = self.hyperparams.model_modeling_answer_dropout
        answer_modeling_forget_bias = self.hyperparams.model_modeling_answer_forget_bias
        answer_modeling_residual_connect = self.hyperparams.model_modeling_answer_residual_connect
        answer_modeling_trainable = self.hyperparams.model_modeling_answer_trainable
        
        with tf.variable_scope("modeling", reuse=tf.AUTO_REUSE):
            answer_modeling, answer_modeling_mask = self._build_fusion_result([answer_interaction], [answer_interaction_mask],
                answer_interaction_unit_dim, answer_modeling_unit_dim, "pass", 0, None, answer_modeling_trainable)
            
            answer_modeling_layer = create_recurrent_layer("bi", answer_modeling_num_layer,
                answer_modeling_unit_dim, answer_modeling_cell_type, answer_modeling_hidden_activation,
                answer_modeling_dropout, answer_modeling_forget_bias, answer_modeling_residual_connect,
                self.num_gpus, self.default_gpu_id, answer_modeling_trainable)
            
            answer_modeling, _ = answer_modeling_layer(answer_modeling, answer_modeling_mask)
        
        return answer_modeling, answer_modeling_mask
    
    def _build_output_layer(self,
                            answer_modeling,
                            answer_modeling_mask):
        """build output layer for bidaf model"""
        answer_modeling_unit_dim = self.hyperparams.model_modeling_answer_unit_dim
        answer_start_num_layer = self.hyperparams.model_output_answer_start_num_layer
        answer_start_unit_dim = self.hyperparams.model_output_answer_start_unit_dim
        answer_start_cell_type = self.hyperparams.model_output_answer_start_cell_type
        answer_start_hidden_activation = self.hyperparams.model_output_answer_start_hidden_activation
        answer_start_dropout = self.hyperparams.model_output_answer_start_dropout
        answer_start_forget_bias = self.hyperparams.model_output_answer_start_forget_bias
        answer_start_residual_connect = self.hyperparams.model_output_answer_start_residual_connect
        answer_start_trainable = self.hyperparams.model_output_answer_start_trainable
        answer_end_num_layer = self.hyperparams.model_output_answer_end_num_layer
        answer_end_unit_dim = self.hyperparams.model_output_answer_end_unit_dim
        answer_end_cell_type = self.hyperparams.model_output_answer_end_cell_type
        answer_end_hidden_activation = self.hyperparams.model_output_answer_end_hidden_activation
        answer_end_dropout = self.hyperparams.model_output_answer_end_dropout
        answer_end_forget_bias = self.hyperparams.model_output_answer_end_forget_bias
        answer_end_residual_connect = self.hyperparams.model_output_answer_end_residual_connect
        answer_end_trainable = self.hyperparams.model_output_answer_end_trainable
        
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            answer_intermediate_list = [answer_modeling]
            answer_intermediate_mask_list = [answer_modeling_mask]
            answer_intermediate_unit_dim = answer_modeling_unit_dim
            
            with tf.variable_scope("start", reuse=tf.AUTO_REUSE):
                answer_start, answer_start_mask = self._build_fusion_result(answer_intermediate_list, answer_intermediate_mask_list,
                    answer_intermediate_unit_dim, answer_start_unit_dim, "pass", 0, None, answer_start_trainable)
                
                answer_start_layer = create_recurrent_layer("bi", answer_start_num_layer,
                    answer_start_unit_dim, answer_start_cell_type, answer_start_hidden_activation,
                    answer_start_dropout, answer_start_forget_bias, answer_start_residual_connect,
                    self.num_gpus, self.default_gpu_id, answer_start_trainable)
                answer_start, _ = answer_start_layer(answer_start, answer_start_mask)
                
                answer_start_output_layer = tf.layers.Dense(units=1, activation=None, trainable=answer_start_trainable)
                answer_start_output = answer_start_output_layer(answer_start)
                answer_start_output_mask = answer_start_mask
            
            answer_intermediate_list.append(answer_start)
            answer_intermediate_mask_list.append(answer_start_mask)
            answer_intermediate_unit_dim = answer_intermediate_unit_dim + answer_start_unit_dim * 2
            
            with tf.variable_scope("end", reuse=tf.AUTO_REUSE):
                answer_end, answer_end_mask = self._build_fusion_result(answer_intermediate_list, answer_intermediate_mask_list,
                    answer_intermediate_unit_dim, answer_end_unit_dim, "pass", 0, None, answer_end_trainable)
                
                answer_end_layer = create_recurrent_layer("bi", answer_end_num_layer,
                    answer_end_unit_dim, answer_end_cell_type, answer_end_hidden_activation,
                    answer_end_dropout, answer_end_forget_bias, answer_end_residual_connect,
                    self.num_gpus, self.default_gpu_id, answer_end_trainable)
                answer_end, _ = answer_end_layer(answer_end, answer_end_mask)
                
                answer_end_output_layer = tf.layers.Dense(units=1, activation=None, trainable=answer_end_trainable)
                answer_end_output = answer_end_output_layer(answer_end)
                answer_end_output_mask = answer_end_mask
        
        return answer_start_output, answer_end_output, answer_start_output_mask, answer_end_output_mask
    
    def _build_graph(self,
                     question_word,
                     question_word_mask,
                     question_subword,
                     question_subword_mask,
                     question_char,
                     question_char_mask,
                     context_word,
                     context_word_mask,
                     context_subword,
                     context_subword_mask,
                     context_char,
                     context_char_mask):
        """build graph for bidaf model"""
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            """build representation layer for bidaf model"""
            self.logger.log_print("# build question representation layer for bidaf model")
            (question_feat, question_feat_mask,
                question_word_embedding_placeholder) = self._build_representation_layer(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask)
            self.logger.log_print("# build context representation layer for bidaf model")
            (context_feat, context_feat_mask,
                context_word_embedding_placeholder) = self._build_representation_layer(context_word, context_word_mask,
                    context_subword, context_subword_mask, context_char, context_char_mask)
            
            """build understanding layer for bidaf model"""
            (question_understanding, context_understanding, question_understanding_mask,
                context_understanding_mask) = self._build_understanding_layer(question_feat,
                    context_feat, question_feat_mask, context_feat_mask)
            
            """build interaction layer for bidaf model"""
            answer_interaction, answer_interaction_mask = self._build_interaction_layer(question_understanding,
                context_understanding, question_understanding_mask, context_understanding_mask)
            
            """build modeling layer for bidaf model"""
            self.logger.log_print("# build answer modeling layer for bidaf model")
            answer_modeling, answer_modeling_mask = self._build_modeling_layer(answer_interaction, answer_interaction_mask)
            
            """build output layer for bidaf model"""
            self.logger.log_print("# build answer output layer for bidaf model")
            (answer_start_output, answer_end_output, answer_start_output_mask,
                answer_end_output_mask) = self._build_output_layer(answer_modeling, answer_modeling_mask)
            
        return (answer_start_output, answer_end_output, answer_start_output_mask, answer_end_output_mask, 
            question_word_embedding_placeholder, context_word_embedding_placeholder)
    
    def _compute_loss(self,
                      label,
                      logit,
                      logit_mask):
        """compute optimization loss"""
        label = tf.squeeze(label)
        logit = tf.squeeze(logit * logit_mask)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
        loss = tf.reduce_sum(cross_entropy) / tf.to_float(self.batch_size)
        
        return loss
    
    def train(self,
              sess,
              question_word_embedding,
              context_word_embedding):
        """train bidaf model"""
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        
        if word_embed_pretrained == True:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.decayed_learning_rate, self.global_step, self.batch_size, self.train_summary],
                feed_dict={self.question_word_embedding_placeholder: question_word_embedding,
                    self.context_word_embedding_placeholder: context_word_embedding})
        else:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.decayed_learning_rate, self.global_step, self.batch_size, self.train_summary])
        
        return TrainResult(loss=loss, learning_rate=learning_rate,
            global_step=global_step, batch_size=batch_size, summary=summary)
    
    def infer(self,
              sess,
              question_word_embedding,
              context_word_embedding):
        """infer bidaf model"""
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        
        if word_embed_pretrained == True:
            (answer_start, answer_end, answer_start_mask, answer_end_mask,
                batch_size, summary) = sess.run([self.infer_answer_start, self.infer_answer_end,
                    self.infer_answer_start_mask, self.infer_answer_end_mask, self.batch_size, self.infer_summary],
                    feed_dict={self.question_word_embedding_placeholder: question_word_embedding,
                        self.context_word_embedding_placeholder: context_word_embedding})
        else:
            (answer_start, answer_end, answer_start_mask, answer_end_mask,
                batch_size, summary) = sess.run([self.infer_answer_start, self.infer_answer_end,
                    self.infer_answer_start_mask, self.infer_answer_end_mask, self.batch_size, self.infer_summary])
        
        max_length = self.hyperparams.data_max_context_length
        
        predict = np.full((batch_size, 2), -1)
        for k in range(batch_size):
            curr_max_value = np.full((max_length), 0.0)
            curr_max_span = np.full((max_length, 2), -1)
            
            start_max_length = np.count_nonzero(answer_start_mask[k, :])
            for i in range(start_max_length):
                if i == 0:
                    curr_max_value[i] = answer_start[k, i]
                    curr_max_span[i, 0] = i
                else:
                    if answer_start[k, i] < curr_max_value[i-1]:
                        curr_max_value[i] = curr_max_value[i-1]
                        curr_max_span[i, 0] = curr_max_span[i-1, 0]
                    else:
                        curr_max_value[i] = answer_start[k, i]
                        curr_max_span[i, 0] = i
            
            end_max_length = np.count_nonzero(answer_end_mask[k, :])
            for j in range(end_max_length):
                curr_max_value[j] = curr_max_value[j] + answer_end[k, j]
                curr_max_span[j, 1] = j
            
            index = np.argmax(curr_max_value)
            predict[k, :] = curr_max_span[index, :]
        
        return InferResult(predict=predict, batch_size=batch_size, summary=summary)            
    
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
