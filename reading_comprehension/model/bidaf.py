import collections
import os.path

import numpy as np
import tensorflow as tf

from util.default_util import *
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
            answer_result = tf.squeeze(self.data_pipeline.input_answer, axis=-1)
            answer_result_mask = tf.squeeze(self.data_pipeline.input_answer_mask, axis=-1)
            
            """build graph for bidaf model"""
            self.logger.log_print("# build graph")
            (answer_start_output, answer_end_output, answer_start_output_mask,
                answer_end_output_mask) = self._build_graph(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask,
                    context_word, context_word_mask, context_subword, context_subword_mask, context_char, context_char_mask)
            answer_start_output_mask = tf.squeeze(answer_start_output_mask, axis=-1)
            answer_end_output_mask = tf.squeeze(answer_end_output_mask, axis=-1)
            answer_start_output = tf.squeeze(answer_start_output, axis=-1)
            answer_end_output = tf.squeeze(answer_end_output, axis=-1)
            self.answer_start_mask = answer_start_output_mask
            self.answer_end_mask = answer_end_output_mask
            self.answer_start = softmax_with_mask(answer_start_output, answer_start_output_mask, axis=-1) * self.answer_start_mask
            self.answer_end = softmax_with_mask(answer_end_output, answer_end_output_mask, axis=-1) * self.answer_end_mask
            
            if self.hyperparams.train_ema_enable == True:
                self.ema = self._get_exponential_moving_average(self.global_step)
                self.variable_list = self.ema.variables_to_restore(tf.trainable_variables())
            else:
                self.variable_list = tf.global_variables()
            
            if self.mode == "infer":
                """get infer answer"""
                self.infer_answer_start_mask = self.answer_start_mask
                self.infer_answer_end_mask = self.answer_end_mask
                self.infer_answer_start = self.answer_start
                self.infer_answer_end = self.answer_end
                
                """create infer summary"""
                self.infer_summary = self._get_infer_summary()
            
            if self.mode == "train":
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                answer_result_mask_shape = tf.shape(answer_result_mask)
                answer_start_result_mask = tf.reshape(answer_result_mask[:,0], shape=[answer_result_mask_shape[0]])
                answer_end_result_mask = tf.reshape(answer_result_mask[:,1], shape=[answer_result_mask_shape[0]])
                answer_result_shape = tf.shape(answer_result)
                answer_start_result = tf.reshape(answer_result[:,0], shape=[answer_result_shape[0]])
                answer_end_result = tf.reshape(answer_result[:,1], shape=[answer_result_shape[0]])
                
                start_loss = self._compute_loss(answer_start_result, answer_start_result_mask,
                    answer_start_output, answer_start_output_mask, self.hyperparams.train_label_smoothing)
                end_loss = self._compute_loss(answer_end_result, answer_end_result_mask,
                    answer_end_output, answer_end_output_mask, self.hyperparams.train_label_smoothing)
                self.train_loss = tf.reduce_mean(start_loss + end_loss)
                
                if self.hyperparams.train_regularization_enable == True:
                    regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    regularization_loss = tf.contrib.layers.apply_regularization(self.regularizer, regularization_variables)
                    self.train_loss = self.train_loss + regularization_loss
                
                """apply learning rate warm-up & decay"""
                self.initial_learning_rate = tf.constant(self.hyperparams.train_optimizer_learning_rate)
                
                if self.hyperparams.train_optimizer_warmup_enable == True:
                    self.logger.log_print("# setup learning rate warm-up mechanism")
                    self.warmup_learning_rate = self._apply_learning_rate_warmup(self.initial_learning_rate)
                else:
                    self.warmup_learning_rate = self.initial_learning_rate
                
                if self.hyperparams.train_optimizer_decay_enable == True:
                    self.logger.log_print("# setup learning rate decay mechanism")
                    self.decayed_learning_rate = self._apply_learning_rate_decay(self.warmup_learning_rate)
                else:
                    self.decayed_learning_rate = self.warmup_learning_rate
                
                self.learning_rate = self.decayed_learning_rate
                
                """initialize optimizer"""
                self.logger.log_print("# initialize optimizer")
                self.optimizer = self._initialize_optimizer(self.decayed_learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.opt_op, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                if self.hyperparams.train_ema_enable == True:
                    with tf.control_dependencies([self.opt_op]):
                        self.update_op = self.ema.apply(tf.trainable_variables())
                else:
                    self.update_op = self.opt_op
                
                """create train summary"""                
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_debug_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "debug")
            self.ckpt_epoch_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "epoch")
            
            if not tf.gfile.Exists(self.ckpt_debug_dir):
                tf.gfile.MakeDirs(self.ckpt_debug_dir)
            
            if not tf.gfile.Exists(self.ckpt_epoch_dir):
                tf.gfile.MakeDirs(self.ckpt_epoch_dir)
            
            self.ckpt_debug_name = os.path.join(self.ckpt_debug_dir, "model_debug_ckpt")
            self.ckpt_epoch_name = os.path.join(self.ckpt_epoch_dir, "model_epoch_ckpt")
            
            if self.mode == "infer":
                self.ckpt_debug_saver = tf.train.Saver(self.variable_list)
                self.ckpt_epoch_saver = tf.train.Saver(self.variable_list, max_to_keep=self.hyperparams.train_num_epoch)  
            
            if self.mode == "train":
                self.ckpt_debug_saver = tf.train.Saver()
                self.ckpt_epoch_saver = tf.train.Saver(max_to_keep=self.hyperparams.train_num_epoch) 
    
    def _build_representation_layer(self,
                                    input_question_word,
                                    input_question_word_mask,
                                    input_question_subword,
                                    input_question_subword_mask,
                                    input_question_char,
                                    input_question_char_mask,
                                    input_context_word,
                                    input_context_word_mask,
                                    input_context_subword,
                                    input_context_subword_mask,
                                    input_context_char,
                                    input_context_char_mask):
        """build representation layer for bidaf model"""
        word_vocab_size = self.hyperparams.data_word_vocab_size
        word_embed_dim = self.hyperparams.model_representation_word_embed_dim
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        word_feat_trainable = self.hyperparams.model_representation_word_feat_trainable
        word_feat_enable = self.hyperparams.model_representation_word_feat_enable
        subword_vocab_size = self.hyperparams.data_subword_vocab_size
        subword_embed_dim = self.hyperparams.model_representation_subword_embed_dim
        subword_unit_dim = self.hyperparams.model_representation_subword_unit_dim
        subword_feat_trainable = self.hyperparams.model_representation_subword_feat_trainable
        subword_window_size = self.hyperparams.model_representation_subword_window_size
        subword_hidden_activation = self.hyperparams.model_representation_subword_hidden_activation
        subword_dropout = self.hyperparams.model_representation_subword_dropout if self.mode == "train" else 0.0
        subword_pooling_type = self.hyperparams.model_representation_subword_pooling_type
        subword_feat_enable = self.hyperparams.model_representation_subword_feat_enable
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_unit_dim = self.hyperparams.model_representation_char_unit_dim
        char_feat_trainable = self.hyperparams.model_representation_char_feat_trainable
        char_window_size = self.hyperparams.model_representation_char_window_size
        char_hidden_activation = self.hyperparams.model_representation_char_hidden_activation
        char_dropout = self.hyperparams.model_representation_char_dropout if self.mode == "train" else 0.0
        char_pooling_type = self.hyperparams.model_representation_char_pooling_type
        char_feat_enable = self.hyperparams.model_representation_char_feat_enable
        fusion_type = self.hyperparams.model_representation_fusion_type
        fusion_num_layer = self.hyperparams.model_representation_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_representation_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_representation_fusion_hidden_activation
        fusion_dropout = self.hyperparams.model_representation_fusion_dropout if self.mode == "train" else 0.0
        fusion_trainable = self.hyperparams.model_representation_fusion_trainable
        default_representation_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            input_question_feat_list = []
            input_question_feat_mask_list = []
            input_context_feat_list = []
            input_context_feat_mask_list = []
            
            if word_feat_enable == True:
                self.logger.log_print("# build word-level representation layer")
                word_feat_layer = WordFeat(vocab_size=word_vocab_size, embed_dim=word_embed_dim,
                    pretrained=word_embed_pretrained, num_gpus=self.num_gpus, default_gpu_id=default_representation_gpu_id,
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=word_feat_trainable)
                
                (input_question_word_feat,
                    input_question_word_feat_mask) = word_feat_layer(input_question_word, input_question_word_mask)
                (input_context_word_feat,
                    input_context_word_feat_mask) = word_feat_layer(input_context_word, input_context_word_mask)
                
                input_question_feat_list.append(input_question_word_feat)
                input_question_feat_mask_list.append(input_question_word_feat_mask)
                input_context_feat_list.append(input_context_word_feat)
                input_context_feat_mask_list.append(input_context_word_feat_mask)
                
                word_unit_dim = word_embed_dim
                self.word_embedding_placeholder = word_feat_layer.get_embedding_placeholder()
            else:
                word_unit_dim = 0
                self.word_embedding_placeholder = None
            
            if subword_feat_enable == True:
                self.logger.log_print("# build subword-level representation layer")
                subword_feat_layer = SubwordFeat(vocab_size=subword_vocab_size, embed_dim=subword_embed_dim,
                    unit_dim=subword_unit_dim, window_size=subword_window_size, hidden_activation=subword_hidden_activation,
                    pooling_type=subword_pooling_type, dropout=subword_dropout, num_gpus=self.num_gpus,
                    default_gpu_id=default_representation_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=subword_feat_trainable)
                
                (input_question_subword_feat,
                    input_question_subword_feat_mask) = subword_feat_layer(input_question_subword, input_question_subword_mask)
                (input_context_subword_feat,
                    input_context_subword_feat_mask) = subword_feat_layer(input_context_subword, input_context_subword_mask)
                
                input_question_feat_list.append(input_question_subword_feat)
                input_question_feat_mask_list.append(input_question_subword_feat_mask)
                input_context_feat_list.append(input_context_subword_feat)
                input_context_feat_mask_list.append(input_context_subword_feat_mask)
            else:
                subword_unit_dim = 0
            
            if char_feat_enable == True:
                self.logger.log_print("# build char-level representation layer")
                char_feat_layer = CharFeat(vocab_size=char_vocab_size, embed_dim=char_embed_dim,
                    unit_dim=char_unit_dim, window_size=char_window_size, hidden_activation=char_hidden_activation,
                    pooling_type=char_pooling_type, dropout=char_dropout, num_gpus=self.num_gpus,
                    default_gpu_id=default_representation_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=char_feat_trainable)
                
                (input_question_char_feat,
                    input_question_char_feat_mask) = char_feat_layer(input_question_char, input_question_char_mask)
                (input_context_char_feat,
                    input_context_char_feat_mask) = char_feat_layer(input_context_char, input_context_char_mask)
                
                input_question_feat_list.append(input_question_char_feat)
                input_question_feat_mask_list.append(input_question_char_feat_mask)
                input_context_feat_list.append(input_context_char_feat)
                input_context_feat_mask_list.append(input_context_char_feat_mask)
            else:
                char_unit_dim = 0
            
            feat_unit_dim = word_unit_dim + subword_unit_dim + char_unit_dim
            feat_fusion_layer = self._create_fusion_layer(feat_unit_dim, fusion_unit_dim,
                fusion_type, fusion_num_layer, fusion_hidden_activation, fusion_dropout,
                self.num_gpus, default_representation_gpu_id, self.regularizer, self.random_seed, fusion_trainable)
            
            input_question_feat, input_question_feat_mask = self._build_fusion_result(input_question_feat_list,
                input_question_feat_mask_list, feat_fusion_layer)
            input_context_feat, input_context_feat_mask = self._build_fusion_result(input_context_feat_list,
                input_context_feat_mask_list, feat_fusion_layer)
        
        return input_question_feat, input_question_feat_mask, input_context_feat, input_context_feat_mask
    
    def _build_understanding_layer(self,
                                   question_feat,
                                   context_feat,
                                   question_feat_mask,
                                   context_feat_mask):
        """build understanding layer for bidaf model"""
        question_understanding_num_layer = self.hyperparams.model_understanding_question_num_layer
        question_understanding_unit_dim = self.hyperparams.model_understanding_question_unit_dim
        question_understanding_cell_type = self.hyperparams.model_understanding_question_cell_type
        question_understanding_hidden_activation = self.hyperparams.model_understanding_question_hidden_activation
        question_understanding_dropout = self.hyperparams.model_understanding_question_dropout if self.mode == "train" else 0.0
        question_understanding_forget_bias = self.hyperparams.model_understanding_question_forget_bias
        question_understanding_residual_connect = self.hyperparams.model_understanding_question_residual_connect
        question_understanding_trainable = self.hyperparams.model_understanding_question_trainable
        context_understanding_num_layer = self.hyperparams.model_understanding_context_num_layer
        context_understanding_unit_dim = self.hyperparams.model_understanding_context_unit_dim
        context_understanding_cell_type = self.hyperparams.model_understanding_context_cell_type
        context_understanding_hidden_activation = self.hyperparams.model_understanding_context_hidden_activation
        context_understanding_dropout = self.hyperparams.model_understanding_context_dropout if self.mode == "train" else 0.0
        context_understanding_forget_bias = self.hyperparams.model_understanding_context_forget_bias
        context_understanding_residual_connect = self.hyperparams.model_understanding_context_residual_connect
        context_understanding_trainable = self.hyperparams.model_understanding_context_trainable
        enable_understanding_sharing = self.hyperparams.model_understanding_enable_sharing
        default_understanding_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("understanding", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("question", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build question understanding layer")
                question_understanding_layer = create_recurrent_layer("bi", question_understanding_num_layer,
                    question_understanding_unit_dim, question_understanding_cell_type, question_understanding_hidden_activation,
                    question_understanding_dropout, question_understanding_forget_bias, question_understanding_residual_connect,
                    None, self.num_gpus, default_understanding_gpu_id, self.random_seed, question_understanding_trainable)
                
                (question_understanding, question_understanding_mask,
                    _, _) = question_understanding_layer(question_feat, question_feat_mask)
            
            with tf.variable_scope("context", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build context understanding layer")
                if enable_understanding_sharing == True:
                    context_understanding_layer = question_understanding_layer
                else:
                    context_understanding_layer = create_recurrent_layer("bi", context_understanding_num_layer,
                        context_understanding_unit_dim, context_understanding_cell_type, context_understanding_hidden_activation,
                        context_understanding_dropout, context_understanding_forget_bias, context_understanding_residual_connect,
                        None, self.num_gpus, default_understanding_gpu_id, self.random_seed, context_understanding_trainable)
                
                (context_understanding, context_understanding_mask,
                    _, _) = context_understanding_layer(context_feat, context_feat_mask)
        
        return question_understanding, context_understanding, question_understanding_mask, context_understanding_mask
    
    def _build_interaction_layer(self,
                                 question_understanding,
                                 context_understanding,
                                 question_understanding_mask,
                                 context_understanding_mask):
        """build interaction layer for bidaf model"""
        question_understanding_unit_dim = self.hyperparams.model_understanding_question_unit_dim * 2
        context_understanding_unit_dim = self.hyperparams.model_understanding_context_unit_dim * 2
        question2context_interaction_attention_dim = self.hyperparams.model_interaction_question2context_attention_dim
        question2context_interaction_score_type = self.hyperparams.model_interaction_question2context_score_type
        question2context_interaction_dropout = self.hyperparams.model_interaction_question2context_dropout if self.mode == "train" else 0.0
        question2context_interaction_att_dropout = self.hyperparams.model_interaction_question2context_attention_dropout if self.mode == "train" else 0.0
        question2context_interaction_trainable = self.hyperparams.model_interaction_question2context_trainable
        question2context_interaction_enable = self.hyperparams.model_interaction_question2context_enable
        context2question_interaction_attention_dim = self.hyperparams.model_interaction_context2question_attention_dim
        context2question_interaction_score_type = self.hyperparams.model_interaction_context2question_score_type
        context2question_interaction_dropout = self.hyperparams.model_interaction_context2question_dropout if self.mode == "train" else 0.0
        context2question_interaction_att_dropout = self.hyperparams.model_interaction_context2question_attention_dropout if self.mode == "train" else 0.0
        context2question_interaction_trainable = self.hyperparams.model_interaction_context2question_trainable
        context2question_interaction_enable = self.hyperparams.model_interaction_context2question_enable
        fusion_type = self.hyperparams.model_interaction_fusion_type
        fusion_num_layer = self.hyperparams.model_interaction_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_interaction_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_interaction_fusion_hidden_activation
        fusion_dropout = self.hyperparams.model_interaction_fusion_dropout if self.mode == "train" else 0.0
        fusion_trainable = self.hyperparams.model_interaction_fusion_trainable
        fusion_combo_enable = self.hyperparams.model_interaction_fusion_combo_enable
        enable_interaction_sharing = self.hyperparams.model_interaction_enable_sharing
        default_interaction_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("interaction", reuse=tf.AUTO_REUSE):
            answer_intermediate_list = [context_understanding]
            answer_intermediate_mask_list = [context_understanding_mask]
            answer_intermediate_unit_dim = context_understanding_unit_dim
            
            attention_matrix = None
            with tf.variable_scope("context2question", reuse=tf.AUTO_REUSE):
                if context2question_interaction_enable == True:
                    self.logger.log_print("# build context2question interaction layer")
                    context2question_attention_layer = create_attention_layer("att",
                        context_understanding_unit_dim, question_understanding_unit_dim,
                        context2question_interaction_attention_dim, -1, context2question_interaction_score_type,
                        context2question_interaction_dropout, context2question_interaction_att_dropout, 0.0,
                        False, False, False, attention_matrix, self.num_gpus, default_interaction_gpu_id,
                        self.regularizer, self.random_seed, context2question_interaction_trainable)
                    
                    if enable_interaction_sharing == True:
                        attention_matrix = context2question_attention_layer.get_attention_matrix()
                    
                    (context2question_interaction, context2question_interaction_mask,
                        _, _) = context2question_attention_layer(context_understanding,
                            question_understanding, context_understanding_mask, question_understanding_mask)
                    
                    answer_intermediate_list.append(context2question_interaction)
                    answer_intermediate_mask_list.append(context2question_interaction_mask)
                    answer_intermediate_unit_dim = answer_intermediate_unit_dim + question_understanding_unit_dim
                    
                    if fusion_combo_enable == True:
                        if question_understanding_unit_dim == context_understanding_unit_dim:
                            context2question_combo = context_understanding * context2question_interaction
                            context2question_combo_mask = context_understanding_mask * context2question_interaction_mask
                            answer_intermediate_list.append(context2question_combo)
                            answer_intermediate_mask_list.append(context2question_combo_mask)
                            answer_intermediate_unit_dim = answer_intermediate_unit_dim + question_understanding_unit_dim
            
            with tf.variable_scope("question2context", reuse=tf.AUTO_REUSE):
                if question2context_interaction_enable == True:
                    self.logger.log_print("# build question2context interaction layer")
                    question2context_attention_layer = create_attention_layer("max_att",
                        context_understanding_unit_dim, question_understanding_unit_dim,
                        question2context_interaction_attention_dim, -1, question2context_interaction_score_type,
                        question2context_interaction_dropout, question2context_interaction_att_dropout, 0.0,
                        False, False, False, attention_matrix, self.num_gpus, default_interaction_gpu_id,
                        self.regularizer, self.random_seed, question2context_interaction_trainable)
                    
                    (question2context_interaction,
                        question2context_interaction_mask) = question2context_attention_layer(context_understanding,
                            question_understanding, context_understanding_mask, question_understanding_mask)
                    
                    if fusion_combo_enable == True:
                        question2context_combo = context_understanding * question2context_interaction
                        question2context_combo_mask = context_understanding_mask * question2context_interaction_mask
                        answer_intermediate_list.append(question2context_combo)
                        answer_intermediate_mask_list.append(question2context_combo_mask)
                        answer_intermediate_unit_dim = answer_intermediate_unit_dim + context_understanding_unit_dim
                    else:
                        answer_intermediate_list.append(question2context_interaction)
                        answer_intermediate_mask_list.append(question2context_interaction_mask)
                        answer_intermediate_unit_dim = answer_intermediate_unit_dim + context_understanding_unit_dim
            
            answer_interaction_fusion_layer = self._create_fusion_layer(answer_intermediate_unit_dim,
                fusion_unit_dim, fusion_type, fusion_num_layer, fusion_hidden_activation, fusion_dropout,
                self.num_gpus, default_interaction_gpu_id, self.regularizer, self.random_seed, fusion_trainable)
            answer_interaction, answer_interaction_mask = self._build_fusion_result(answer_intermediate_list,
                answer_intermediate_mask_list, answer_interaction_fusion_layer)
        
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
        answer_modeling_dropout = self.hyperparams.model_modeling_answer_dropout if self.mode == "train" else 0.0
        answer_modeling_att_dropout = self.hyperparams.model_modeling_answer_attention_dropout if self.mode == "train" else 0.0
        answer_modeling_forget_bias = self.hyperparams.model_modeling_answer_forget_bias
        answer_modeling_residual_connect = self.hyperparams.model_modeling_answer_residual_connect
        answer_modeling_attention_dim = self.hyperparams.model_modeling_answer_attention_dim
        answer_modeling_score_type = self.hyperparams.model_modeling_answer_score_type
        answer_modeling_attention_enable = self.hyperparams.model_modeling_answer_attention_enable
        answer_modeling_trainable = self.hyperparams.model_modeling_answer_trainable
        fusion_type = self.hyperparams.model_modeling_fusion_type
        fusion_num_layer = self.hyperparams.model_modeling_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_modeling_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_modeling_fusion_hidden_activation
        fusion_dropout = self.hyperparams.model_modeling_fusion_dropout if self.mode == "train" else 0.0
        fusion_trainable = self.hyperparams.model_modeling_fusion_trainable
        default_modeling_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("modeling", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build answer modeling layer")
            answer_modeling_sequence_layer = create_recurrent_layer("bi", answer_modeling_num_layer,
                answer_modeling_unit_dim, answer_modeling_cell_type, answer_modeling_hidden_activation,
                answer_modeling_dropout, answer_modeling_forget_bias, answer_modeling_residual_connect,
                None, self.num_gpus, default_modeling_gpu_id, self.random_seed, answer_modeling_trainable)
            
            (answer_modeling_sequence, answer_modeling_sequence_mask,
                _, _) = answer_modeling_sequence_layer(answer_interaction, answer_interaction_mask)
            answer_modeling_sequence_unit_dim = answer_modeling_unit_dim * 2
            
            answer_intermediate_list = [answer_interaction]
            answer_intermediate_mask_list = [answer_interaction_mask]
            answer_intermediate_unit_dim = answer_interaction_unit_dim
            if answer_modeling_attention_enable == True:
                answer_modeling_attention_layer = create_attention_layer("att",
                    answer_modeling_sequence_unit_dim, answer_modeling_sequence_unit_dim,
                    answer_modeling_attention_dim, -1, answer_modeling_score_type,
                    answer_modeling_dropout, answer_modeling_att_dropout, 0.0, False, False, True, None,
                    self.num_gpus, default_modeling_gpu_id, self.regularizer, self.random_seed, answer_modeling_trainable)

                (answer_modeling_attention, answer_modeling_attention_mask,
                    _, _) = answer_modeling_attention_layer(answer_modeling_sequence,
                        answer_modeling_sequence, answer_modeling_sequence_mask, answer_modeling_sequence_mask)
                answer_modeling_attention_unit_dim = answer_modeling_sequence_unit_dim
                
                answer_intermediate_list.append(answer_modeling_attention)
                answer_intermediate_mask_list.append(answer_modeling_attention_mask)
                answer_intermediate_unit_dim = answer_intermediate_unit_dim + answer_modeling_attention_unit_dim
            else:
                answer_intermediate_list.append(answer_modeling_sequence)
                answer_intermediate_mask_list.append(answer_modeling_sequence_mask)
                answer_intermediate_unit_dim = answer_intermediate_unit_dim + answer_modeling_sequence_unit_dim
            
            answer_modeling_fusion_layer = self._create_fusion_layer(answer_intermediate_unit_dim,
                fusion_unit_dim, fusion_type, fusion_num_layer, fusion_hidden_activation, fusion_dropout,
                self.num_gpus, default_modeling_gpu_id, self.regularizer, self.random_seed, fusion_trainable)
            answer_modeling, answer_modeling_mask = self._build_fusion_result(answer_intermediate_list,
                answer_intermediate_mask_list, answer_modeling_fusion_layer)
        
        return answer_modeling, answer_modeling_mask
    
    def _build_output_layer(self,
                            answer_modeling,
                            answer_modeling_mask):
        """build output layer for bidaf model"""
        answer_modeling_unit_dim = self.hyperparams.model_modeling_fusion_unit_dim
        answer_start_num_layer = self.hyperparams.model_output_answer_start_num_layer
        answer_start_unit_dim = self.hyperparams.model_output_answer_start_unit_dim
        answer_start_cell_type = self.hyperparams.model_output_answer_start_cell_type
        answer_start_hidden_activation = self.hyperparams.model_output_answer_start_hidden_activation
        answer_start_dropout = self.hyperparams.model_output_answer_start_dropout if self.mode == "train" else 0.0
        answer_start_forget_bias = self.hyperparams.model_output_answer_start_forget_bias
        answer_start_residual_connect = self.hyperparams.model_output_answer_start_residual_connect
        answer_start_trainable = self.hyperparams.model_output_answer_start_trainable
        answer_end_num_layer = self.hyperparams.model_output_answer_end_num_layer
        answer_end_unit_dim = self.hyperparams.model_output_answer_end_unit_dim
        answer_end_cell_type = self.hyperparams.model_output_answer_end_cell_type
        answer_end_hidden_activation = self.hyperparams.model_output_answer_end_hidden_activation
        answer_end_dropout = self.hyperparams.model_output_answer_end_dropout if self.mode == "train" else 0.0
        answer_end_forget_bias = self.hyperparams.model_output_answer_end_forget_bias
        answer_end_residual_connect = self.hyperparams.model_output_answer_end_residual_connect
        answer_end_trainable = self.hyperparams.model_output_answer_end_trainable
        default_output_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build answer output layer")
            answer_intermediate_list = [answer_modeling]
            answer_intermediate_mask_list = [answer_modeling_mask]
            answer_output_list = []
            answer_output_mask_list = []
            
            with tf.variable_scope("start", reuse=tf.AUTO_REUSE):
                answer_start_layer = create_recurrent_layer("bi", answer_start_num_layer,
                    answer_start_unit_dim, answer_start_cell_type, answer_start_hidden_activation,
                    answer_start_dropout, answer_start_forget_bias, answer_start_residual_connect,
                    None, self.num_gpus, default_output_gpu_id, self.random_seed, answer_start_trainable)
                answer_start, answer_start_mask, _, _ = answer_start_layer(answer_modeling, answer_modeling_mask)
                
                (answer_start_fusion,
                    answer_start_fusion_mask) = self._build_fusion_result([answer_modeling, answer_start],
                        [answer_modeling_mask, answer_start_mask], None)
                
                answer_start_output_layer = create_dense_layer("single", 1, 1, 1, "",
                    [answer_start_dropout], None, False, False, False, self.num_gpus, default_output_gpu_id,
                    self.regularizer, self.random_seed, answer_start_trainable)
                (answer_start_output,
                    answer_start_output_mask) = answer_start_output_layer(answer_start_fusion,
                        answer_start_fusion_mask)
                answer_output_list.append(answer_start_output)
                answer_output_mask_list.append(answer_start_output_mask)
            
            answer_intermediate_list.append(answer_start)
            answer_intermediate_mask_list.append(answer_start_mask)
            (answer_intermediate,
                answer_intermediate_mask) = self._build_fusion_result(answer_intermediate_list,
                    answer_intermediate_mask_list, None)
            
            with tf.variable_scope("end", reuse=tf.AUTO_REUSE):
                answer_end_layer = create_recurrent_layer("bi", answer_end_num_layer,
                    answer_end_unit_dim, answer_end_cell_type, answer_end_hidden_activation,
                    answer_end_dropout, answer_end_forget_bias, answer_end_residual_connect,
                    None, self.num_gpus, default_output_gpu_id, self.random_seed, answer_end_trainable)
                answer_end, answer_end_mask, _, _ = answer_end_layer(answer_intermediate, answer_intermediate_mask)
                
                (answer_end_fusion,
                    answer_end_fusion_mask) = self._build_fusion_result([answer_modeling, answer_end],
                        [answer_modeling_mask, answer_end_mask], None)
                
                answer_end_output_layer = create_dense_layer("single", 1, 1, 1, "",
                    [answer_end_dropout], None, False, False, False, self.num_gpus, default_output_gpu_id,
                    self.regularizer, self.random_seed, answer_end_trainable)
                (answer_end_output,
                    answer_end_output_mask) = answer_end_output_layer(answer_end_fusion,
                        answer_end_fusion_mask)
                answer_output_list.append(answer_end_output)
                answer_output_mask_list.append(answer_end_output_mask)
        
        return answer_output_list, answer_output_mask_list
    
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
            (question_feat, question_feat_mask, context_feat,
                context_feat_mask) = self._build_representation_layer(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask, context_word,
                    context_word_mask, context_subword, context_subword_mask, context_char, context_char_mask)
            
            """build understanding layer for bidaf model"""
            (question_understanding, context_understanding, question_understanding_mask,
                context_understanding_mask) = self._build_understanding_layer(question_feat,
                    context_feat, question_feat_mask, context_feat_mask)
            
            """build interaction layer for bidaf model"""
            answer_interaction, answer_interaction_mask = self._build_interaction_layer(question_understanding,
                context_understanding, question_understanding_mask, context_understanding_mask)
            
            """build modeling layer for bidaf model"""
            answer_modeling, answer_modeling_mask = self._build_modeling_layer(answer_interaction, answer_interaction_mask)
            
            """build output layer for bidaf model"""
            answer_output_list, answer_output_mask_list = self._build_output_layer(answer_modeling, answer_modeling_mask)
            answer_start_output = answer_output_list[0]
            answer_end_output = answer_output_list[1]
            answer_start_output_mask = answer_output_mask_list[0]
            answer_end_output_mask = answer_output_mask_list[1]
            
        return answer_start_output, answer_end_output, answer_start_output_mask, answer_end_output_mask
    
    def _compute_loss(self,
                      label,
                      label_mask,
                      predict,
                      predict_mask,
                      label_smoothing):
        """compute optimization loss"""
        masked_predict = generate_masked_data(predict, predict_mask)
        masked_label = tf.cast(label, dtype=tf.int32) * tf.cast(label_mask, dtype=tf.int32)
        
        if label_smoothing > 1e-10:
            onehot_label = generate_onehot_label(masked_label, tf.shape(masked_predict)[-1])
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_label,
                logits=masked_predict, label_smoothing=label_smoothing)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_label, logits=masked_predict)
        
        return loss
    
    def save(self,
             sess,
             global_step,
             save_mode):
        """save checkpoint for bidaf model"""
        if save_mode == "debug":
            self.ckpt_debug_saver.save(sess, self.ckpt_debug_name, global_step=global_step)
        elif save_mode == "epoch":
            self.ckpt_epoch_saver.save(sess, self.ckpt_epoch_name, global_step=global_step)
        else:
            raise ValueError("unsupported save mode {0}".format(save_mode))
    
    def restore(self,
                sess,
                ckpt_file,
                ckpt_type):
        """restore bidaf model from checkpoint"""
        if ckpt_file is None:
            raise FileNotFoundError("checkpoint file doesn't exist")
        
        if ckpt_type == "debug":
            self.ckpt_debug_saver.restore(sess, ckpt_file)
        elif ckpt_type == "epoch":
            self.ckpt_epoch_saver.restore(sess, ckpt_file)
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_latest_ckpt(self,
                        ckpt_type):
        """get the latest checkpoint for bidaf model"""
        if ckpt_type == "debug":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_debug_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        elif ckpt_type == "epoch":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_epoch_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_ckpt_list(self,
                      ckpt_type):
        """get checkpoint list for bidaf model"""
        if ckpt_type == "debug":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_debug_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        elif ckpt_type == "epoch":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_epoch_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))

class WordFeat(object):
    """word-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 pretrained,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="word_feat"):
        """initialize word-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pretrained = pretrained
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size, self.embed_dim, self.pretrained,
                self.num_gpus, self.default_gpu_id, None, self.random_seed, self.trainable)
    
    def __call__(self,
                 input_word,
                 input_word_mask):
        """call word-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_word_feat_mask = input_word_mask
            input_word_feat = tf.squeeze(self.embedding_layer(input_word), axis=-2)
        
        return input_word_feat, input_word_feat_mask
    
    def get_embedding_placeholder(self):
        """get word-level embedding placeholder"""
        return self.embedding_layer.get_embedding_placeholder()

class SubwordFeat(object):
    """subword-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 unit_dim,
                 window_size,
                 hidden_activation,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="subword_feat"):
        """initialize subword-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.hidden_activation = hidden_activation
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size, self.embed_dim, False,
                self.num_gpus, self.default_gpu_id, None, self.random_seed, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id, self.random_seed)
            
            self.conv_layer = create_convolution_layer("multi_1d", 1, self.embed_dim,
                self.unit_dim, 1, self.window_size, 1, "SAME", self.hidden_activation, [0.0], None,
                False, False, True, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_subword,
                 input_subword_mask):
        """call subword-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_subword_embedding_mask = tf.expand_dims(input_subword_mask, axis=-1)
            input_subword_embedding = self.embedding_layer(input_subword)
            
            (input_subword_dropout,
                input_subword_dropout_mask) = self.dropout_layer(input_subword_embedding, input_subword_embedding_mask)
            
            (input_subword_conv,
                input_subword_conv_mask) = self.conv_layer(input_subword_dropout, input_subword_dropout_mask)
            
            (input_subword_pool,
                input_subword_pool_mask) = self.pooling_layer(input_subword_conv, input_subword_conv_mask)
            
            input_subword_feat = input_subword_pool
            input_subword_feat_mask = input_subword_pool_mask
        
        return input_subword_feat, input_subword_feat_mask

class CharFeat(object):
    """char-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 unit_dim,
                 window_size,
                 hidden_activation,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="char_feat"):
        """initialize char-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.hidden_activation = hidden_activation
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size, self.embed_dim, False,
                self.num_gpus, self.default_gpu_id, None, self.random_seed, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id, self.random_seed)
            
            self.conv_layer = create_convolution_layer("multi_1d", 1, self.embed_dim,
                self.unit_dim, 1, self.window_size, 1, "SAME", self.hidden_activation, [0.0], None,
                False, False, True, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_char,
                 input_char_mask):
        """call char-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_char_embedding_mask = tf.expand_dims(input_char_mask, axis=-1)
            input_char_embedding = self.embedding_layer(input_char)
            
            (input_char_dropout,
                input_char_dropout_mask) = self.dropout_layer(input_char_embedding, input_char_embedding_mask)
            
            (input_char_conv,
                input_char_conv_mask) = self.conv_layer(input_char_dropout, input_char_dropout_mask)
            
            (input_char_pool,
                input_char_pool_mask) = self.pooling_layer(input_char_conv, input_char_conv_mask)
            
            input_char_feat = input_char_pool
            input_char_feat_mask = input_char_pool_mask
        
        return input_char_feat, input_char_feat_mask
