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
            answer_result = self.data_pipeline.input_answer
            answer_result_mask = self.data_pipeline.input_answer_mask
            
            """build graph for bidaf model"""
            self.logger.log_print("# build graph")
            (answer_start_output, answer_end_output, answer_start_output_mask,
                answer_end_output_mask) = self._build_graph(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask,
                    context_word, context_word_mask, context_subword, context_subword_mask, context_char, context_char_mask)
            self.answer_start_mask = tf.squeeze(answer_start_output_mask)
            self.answer_end_mask = tf.squeeze(answer_end_output_mask)
            self.answer_start = softmax_with_mask(tf.squeeze(answer_start_output),
                self.answer_start_mask, axis=-1)
            self.answer_end = softmax_with_mask(tf.squeeze(answer_end_output),
                self.answer_end_mask, axis=-1)
            
            self.variable_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.variable_lookup = {v.op.name: v for v in self.variable_list}
            
            if self.hyperparams.train_ema_enable == True:
                self.ema = tf.train.ExponentialMovingAverage(decay=self.hyperparams.train_ema_decay_rate)
            
            if self.mode == "infer":
                """get infer answer"""
                self.infer_answer_start_mask = self.answer_start_mask
                self.infer_answer_end_mask = self.answer_end_mask
                self.infer_answer_start = self.answer_start
                self.infer_answer_end = self.answer_end
                
                if self.hyperparams.train_ema_enable == True:
                    self.variable_lookup = {self.ema.average_name(v): v for v in self.variable_list}
                
                """create infer summary"""
                self.infer_summary = self._get_infer_summary()
            
            if self.mode == "train":
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                answer_start_result = answer_result[:,0,:]
                answer_end_result = answer_result[:,1,:]
                start_loss = self._compute_loss(answer_start_result, self.answer_start, self.answer_start_mask)
                end_loss = self._compute_loss(answer_end_result, self.answer_end, self.answer_end_mask)
                self.train_loss = start_loss + end_loss
                
                """apply learning rate decay"""
                self.learning_rate = tf.constant(self.hyperparams.train_optimizer_learning_rate)
                
                if self.hyperparams.train_optimizer_decay_enable == True:
                    self.logger.log_print("# setup learning rate decay mechanism")
                    self.decayed_learning_rate = self._apply_learning_rate_decay(self.learning_rate)
                else:
                    self.decayed_learning_rate = self.learning_rate
                
                """initialize optimizer"""
                self.logger.log_print("# initialize optimizer")
                self.optimizer = self._initialize_optimizer(self.decayed_learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.update_model, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                if self.hyperparams.train_ema_enable == True:
                    with tf.control_dependencies([self.update_model]):
                        self.update_op = self.ema.apply(self.variable_list)
                    
                    self.variable_lookup = {self.ema.average_name(v): self.ema.average(v) for v in self.variable_list}
                else:
                    self.update_op = self.update_model
                
                """create train summary"""
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver(self.variable_lookup)
    
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
                    self.num_gpus, default_understanding_gpu_id, True, question_understanding_trainable)
                
                question_understanding, question_understanding_mask = question_understanding_layer(question_feat, question_feat_mask)
            
            with tf.variable_scope("context", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build context understanding layer")
                if enable_understanding_sharing == True:
                    context_understanding_layer = question_understanding_layer
                else:
                    context_understanding_layer = create_recurrent_layer("bi", context_understanding_num_layer,
                        context_understanding_unit_dim, context_understanding_cell_type, context_understanding_hidden_activation,
                        context_understanding_dropout, context_understanding_forget_bias, context_understanding_residual_connect,
                        self.num_gpus, default_understanding_gpu_id, True, context_understanding_trainable)
                
                context_understanding, context_understanding_mask = context_understanding_layer(context_feat, context_feat_mask)
        
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
        question2context_interaction_trainable = self.hyperparams.model_interaction_question2context_trainable
        question2context_interaction_enable = self.hyperparams.model_interaction_question2context_enable
        context2question_interaction_attention_dim = self.hyperparams.model_interaction_context2question_attention_dim
        context2question_interaction_score_type = self.hyperparams.model_interaction_context2question_score_type
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
        default_interaction_gpu_id = self.default_gpu_id + 1
        
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
                        context2question_interaction_attention_dim, context2question_interaction_score_type,
                        False, False, False, attention_matrix, self.num_gpus, default_interaction_gpu_id,
                        True, self.regularizer, context2question_interaction_trainable)
                    
                    if enable_interaction_sharing == True:
                        attention_matrix = context2question_attention_layer.get_attention_matrix()
                    
                    (context2question_interaction,
                        context2question_interaction_mask) = context2question_attention_layer(context_understanding,
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
                        question2context_interaction_attention_dim, question2context_interaction_score_type,
                        False, False, False, attention_matrix, self.num_gpus, default_interaction_gpu_id,
                        True, self.regularizer, question2context_interaction_trainable)
                    
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
                self.num_gpus, default_interaction_gpu_id, self.regularizer, fusion_trainable)
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
        default_modeling_gpu_id = self.default_gpu_id + 2
        
        with tf.variable_scope("modeling", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build answer modeling layer")
            answer_modeling_sequence_layer = create_recurrent_layer("bi", answer_modeling_num_layer,
                answer_modeling_unit_dim, answer_modeling_cell_type, answer_modeling_hidden_activation,
                answer_modeling_dropout, answer_modeling_forget_bias, answer_modeling_residual_connect,
                self.num_gpus, default_modeling_gpu_id, True, answer_modeling_trainable)
            
            (answer_modeling_sequence,
                answer_modeling_sequence_mask) = answer_modeling_sequence_layer(answer_interaction, answer_interaction_mask)
            answer_modeling_sequence_unit_dim = answer_modeling_unit_dim * 2
            
            answer_intermediate_list = [answer_interaction]
            answer_intermediate_mask_list = [answer_interaction_mask]
            answer_intermediate_unit_dim = answer_interaction_unit_dim
            if answer_modeling_attention_enable == True:
                answer_modeling_attention_layer = create_attention_layer("att",
                    answer_modeling_sequence_unit_dim, answer_modeling_sequence_unit_dim,
                    answer_modeling_attention_dim, answer_modeling_score_type, False, False, True,
                    None, self.num_gpus, default_modeling_gpu_id, True, self.regularizer, answer_modeling_trainable)

                (answer_modeling_attention,
                    answer_modeling_attention_mask) = answer_modeling_attention_layer(answer_modeling_sequence,
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
                self.num_gpus, default_modeling_gpu_id, self.regularizer, fusion_trainable)
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
            
            with tf.variable_scope("start", reuse=tf.AUTO_REUSE):
                answer_start_layer = create_recurrent_layer("bi", answer_start_num_layer,
                    answer_start_unit_dim, answer_start_cell_type, answer_start_hidden_activation,
                    answer_start_dropout, answer_start_forget_bias, answer_start_residual_connect,
                    self.num_gpus, default_output_gpu_id, True, answer_start_trainable)
                answer_start, answer_start_mask = answer_start_layer(answer_modeling, answer_modeling_mask)
                
                (answer_start_fusion,
                    answer_start_fusion_mask) = self._build_fusion_result([answer_modeling, answer_start],
                        [answer_modeling_mask, answer_start_mask], None)
                
                answer_start_fusion = tf.nn.dropout(answer_start_fusion, 1.0-answer_start_dropout)
                answer_start_output_layer = create_dense_layer(1, 1, "", 0.0, False, False,
                    self.num_gpus, default_output_gpu_id, True, answer_start_trainable)
                (answer_start_output,
                    answer_start_output_mask) = answer_start_output_layer(answer_start_fusion,
                        answer_start_fusion_mask)
            
            answer_intermediate_list.append(answer_start)
            answer_intermediate_mask_list.append(answer_start_mask)
            (answer_intermediate,
                answer_intermediate_mask) = self._build_fusion_result(answer_intermediate_list,
                    answer_intermediate_mask_list, None)
            
            with tf.variable_scope("end", reuse=tf.AUTO_REUSE):
                answer_end_layer = create_recurrent_layer("bi", answer_end_num_layer,
                    answer_end_unit_dim, answer_end_cell_type, answer_end_hidden_activation,
                    answer_end_dropout, answer_end_forget_bias, answer_end_residual_connect,
                    self.num_gpus, default_output_gpu_id, True, answer_end_trainable)
                answer_end, answer_end_mask = answer_end_layer(answer_intermediate, answer_intermediate_mask)
                
                (answer_end_fusion,
                    answer_end_fusion_mask) = self._build_fusion_result([answer_modeling, answer_end],
                        [answer_modeling_mask, answer_end_mask], None)
                
                answer_end_fusion = tf.nn.dropout(answer_end_fusion, 1.0-answer_end_dropout)
                answer_end_output_layer = create_dense_layer(1, 1, "", 0.0, False, False,
                    self.num_gpus, default_output_gpu_id, True, answer_end_trainable)
                (answer_end_output,
                    answer_end_output_mask) = answer_end_output_layer(answer_end_fusion,
                        answer_end_fusion_mask)
        
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
            (answer_start_output, answer_end_output, answer_start_output_mask,
                answer_end_output_mask) = self._build_output_layer(answer_modeling, answer_modeling_mask)
            
        return answer_start_output, answer_end_output, answer_start_output_mask, answer_end_output_mask
    
    def _compute_loss(self,
                      label,
                      logit,
                      logit_mask):
        """compute optimization loss"""
        logit = tf.squeeze(-1.0 * tf.log(logit * logit_mask + EPSILON) * logit_mask)
        label = tf.one_hot(tf.squeeze(label), depth=tf.shape(logit)[1], on_value=1.0, off_value=0.0, dtype=tf.float32)
        loss = tf.reduce_sum(logit * label) / tf.to_float(self.batch_size)
        
        return loss
    
    def train(self,
              sess,
              word_embedding):
        """train bidaf model"""
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        
        if word_embed_pretrained == True:
            (_, loss, learning_rate, global_step, batch_size, summary) = sess.run([self.update_op,
                self.train_loss, self.decayed_learning_rate, self.global_step, self.batch_size, self.train_summary],
                feed_dict={self.word_embedding_placeholder: word_embedding})
        else:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_op,
                self.train_loss, self.decayed_learning_rate, self.global_step, self.batch_size, self.train_summary])
        
        return TrainResult(loss=loss, learning_rate=learning_rate,
            global_step=global_step, batch_size=batch_size, summary=summary)
    
    def infer(self,
              sess,
              word_embedding):
        """infer bidaf model"""
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        
        if word_embed_pretrained == True:
            (answer_start, answer_end, answer_start_mask, answer_end_mask,
                batch_size, summary) = sess.run([self.infer_answer_start, self.infer_answer_end,
                    self.infer_answer_start_mask, self.infer_answer_end_mask, self.batch_size, self.infer_summary],
                    feed_dict={self.word_embedding_placeholder: word_embedding})
        else:
            (answer_start, answer_end, answer_start_mask, answer_end_mask,
                batch_size, summary) = sess.run([self.infer_answer_start, self.infer_answer_end,
                    self.infer_answer_start_mask, self.infer_answer_end_mask, self.batch_size, self.infer_summary])
        
        max_length = self.hyperparams.data_max_context_length
        
        predict = np.full((batch_size, 2), -1)
        for k in range(batch_size):
            curr_max_value = np.full((max_length), float('-inf'))
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
                curr_max_value[j] = curr_max_value[j] * answer_end[k, j]
                curr_max_span[j, 1] = j
            
            index = np.argmax(curr_max_value)
            predict[k, :] = curr_max_span[index, :]
        
        predict_start = np.expand_dims(answer_start, axis=1)
        predict_end = np.expand_dims(answer_end, axis=1)
        predict_detail = np.concatenate((predict_start, predict_end), axis=1)
        
        return InferResult(predict=predict, predict_detail=predict_detail, batch_size=batch_size, summary=summary)            
    
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
