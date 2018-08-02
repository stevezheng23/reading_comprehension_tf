import collections
import os.path

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *
from util.layer_util import *

from model.base_model import *

__all__ = ["RNet"]

class RNet(BaseModel):
    """rnet model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="rnet"):
        """initialize rnet model"""        
        super(RNet, self).__init__(logger=logger, hyperparams=hyperparams,
            data_pipeline=data_pipeline, mode=mode, scope=scope)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer, trainable=False)
            
            """use rnet feature layer"""
            self.word_feat_creator = WordFeat
            self.subword_feat_creator = SubwordFeat
            self.char_feat_creator = CharFeat
            
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
            
            """build graph for rnet model"""
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
                
                if self.hyperparams.train_regularization_enable == True:
                    regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    regularization_loss = tf.contrib.layers.apply_regularization(self.regularizer, regularization_variables)
                    self.train_loss = self.train_loss + regularization_loss
                
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
        """build understanding layer for rnet model"""
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
                    None, self.num_gpus, default_understanding_gpu_id, True, question_understanding_trainable)
                
                question_understanding, question_understanding_mask = question_understanding_layer(question_feat, question_feat_mask)
            
            with tf.variable_scope("context", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build context understanding layer")
                if enable_understanding_sharing == True:
                    context_understanding_layer = question_understanding_layer
                else:
                    context_understanding_layer = create_recurrent_layer("bi", context_understanding_num_layer,
                        context_understanding_unit_dim, context_understanding_cell_type, context_understanding_hidden_activation,
                        context_understanding_dropout, context_understanding_forget_bias, context_understanding_residual_connect,
                        None, self.num_gpus, default_understanding_gpu_id, True, context_understanding_trainable)
                
                context_understanding, context_understanding_mask = context_understanding_layer(context_feat, context_feat_mask)
        
        return question_understanding, context_understanding, question_understanding_mask, context_understanding_mask
    
    def _build_interaction_layer(self,
                                 question_understanding,
                                 context_understanding,
                                 question_understanding_mask,
                                 context_understanding_mask):
        """build interaction layer for rnet model"""
        question_understanding_unit_dim = self.hyperparams.model_understanding_question_unit_dim * 2
        context_understanding_unit_dim = self.hyperparams.model_understanding_context_unit_dim * 2
        context2question_interaction_num_layer = self.hyperparams.model_interaction_context2question_num_layer
        context2question_interaction_unit_dim = self.hyperparams.model_interaction_context2question_unit_dim
        context2question_interaction_cell_type = self.hyperparams.model_interaction_context2question_cell_type
        context2question_interaction_hidden_activation = self.hyperparams.model_interaction_context2question_hidden_activation
        context2question_interaction_dropout = self.hyperparams.model_interaction_context2question_dropout if self.mode == "train" else 0.0
        context2question_interaction_forget_bias = self.hyperparams.model_interaction_context2question_forget_bias
        context2question_interaction_residual_connect = self.hyperparams.model_interaction_context2question_residual_connect
        context2question_interaction_attention_dim = self.hyperparams.model_interaction_context2question_attention_dim
        context2question_interaction_score_type = self.hyperparams.model_interaction_context2question_score_type
        context2question_interaction_trainable = self.hyperparams.model_interaction_context2question_trainable
        default_interaction_gpu_id = self.default_gpu_id + 1
        
        context2question_attention_mechanism = AttentionMechanism(memory=question_understanding,
            memory_mask=question_understanding_mask, attention_type="gated_att",
            src_dim=context_understanding_unit_dim + context2question_interaction_unit_dim,
            trg_dim=question_understanding_unit_dim, att_dim=context2question_interaction_attention_dim,
            score_type=context2question_interaction_score_type, layer_dropout=False, layer_norm=False, residual_connect=False,
            is_self=False, external_matrix=None, num_gpus=self.num_gpus, default_gpu_id=default_interaction_gpu_id,
            regularizer=self.regularizer, trainable=context2question_interaction_trainable)
        
        context2question_interaction_layer = create_recurrent_layer("uni",
            context2question_interaction_num_layer, context2question_interaction_unit_dim,
            context2question_interaction_cell_type, context2question_interaction_hidden_activation,
            context2question_interaction_dropout, context2question_interaction_forget_bias,
            context2question_interaction_residual_connect, context2question_attention_mechanism,
            self.num_gpus, default_interaction_gpu_id, True, context2question_interaction_trainable)
        
        (context2question_interaction,
            context2question_interaction_mask) = context2question_interaction_layer(context_understanding,
                context_understanding_mask)
        answer_interaction = context2question_interaction
        answer_interaction_mask = context2question_interaction_mask
        
        return answer_interaction, answer_interaction_mask
    
    def _build_modeling_layer(self,
                              answer_interaction,
                              answer_interaction_mask):
        """build modeling layer for rnet model"""
        answer_interaction_unit_dim = self.hyperparams.model_interaction_context2question_unit_dim
        answer_modeling_num_layer = self.hyperparams.model_modeling_answer_num_layer
        answer_modeling_unit_dim = self.hyperparams.model_modeling_answer_unit_dim
        answer_modeling_cell_type = self.hyperparams.model_modeling_answer_cell_type
        answer_modeling_hidden_activation = self.hyperparams.model_modeling_answer_hidden_activation
        answer_modeling_dropout = self.hyperparams.model_modeling_answer_dropout if self.mode == "train" else 0.0
        answer_modeling_forget_bias = self.hyperparams.model_modeling_answer_forget_bias
        answer_modeling_residual_connect = self.hyperparams.model_modeling_answer_residual_connect
        answer_modeling_attention_dim = self.hyperparams.model_modeling_answer_attention_dim
        answer_modeling_score_type = self.hyperparams.model_modeling_answer_score_type
        answer_modeling_trainable = self.hyperparams.model_modeling_answer_trainable
        default_modeling_gpu_id = self.default_gpu_id + 2
        
        with tf.variable_scope("modeling", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build answer modeling layer")
            answer_modeling_attention_layer = create_attention_layer("gated_att",
                answer_interaction_unit_dim, answer_interaction_unit_dim,
                answer_modeling_attention_dim, answer_modeling_score_type, 0.0, False, False, True, None,
                self.num_gpus, default_modeling_gpu_id, True, self.regularizer, answer_modeling_trainable)
            
            (answer_modeling_attention,
                answer_modeling_attention_mask) = answer_modeling_attention_layer(answer_interaction,
                    answer_interaction, answer_interaction_mask, answer_interaction_mask)
            
            answer_modeling_sequence_layer = create_recurrent_layer("bi", answer_modeling_num_layer,
                answer_modeling_unit_dim, answer_modeling_cell_type, answer_modeling_hidden_activation,
                answer_modeling_dropout, answer_modeling_forget_bias, answer_modeling_residual_connect,
                None, self.num_gpus, default_modeling_gpu_id, True, answer_modeling_trainable)
            
            (answer_modeling_sequence,
                answer_modeling_sequence_mask) = answer_modeling_sequence_layer(answer_modeling_attention,
                    answer_modeling_attention_mask)
            answer_modeling = answer_modeling_sequence
            answer_modeling_mask = answer_modeling_sequence_mask
        
        return answer_modeling, answer_modeling_mask
    
    def _build_output_layer(self,
                            question_understanding,
                            question_understanding_mask,
                            answer_modeling,
                            answer_modeling_mask):
        def create_base_variable(batch_size,
                                 unit_dim,
                                 num_gpus,
                                 default_gpu_id,
                                 regularizer,
                                 trainable):
            initializer = create_variable_initializer("glorot_uniform")
            base_variable = tf.get_variable("base_variable", shape=[1, 1, unit_dim],
                initializer=initializer, regularizer=regularizer, trainable=trainable, dtype=tf.float32)
            base_variable = tf.tile(base_variable, multiples=[batch_size, 1, 1])
            base_variable_mask = tf.ones([batch_size, 1, 1])
            
            return base_variable, base_variable_mask
        
        """build output layer for rnet model"""
        question_understanding_unit_dim = self.hyperparams.model_understanding_question_unit_dim * 2
        answer_modeling_unit_dim = self.hyperparams.model_modeling_answer_unit_dim * 2
        answer_output_num_layer = self.hyperparams.model_output_answer_num_layer
        answer_output_unit_dim = self.hyperparams.model_output_answer_unit_dim
        answer_output_cell_type = self.hyperparams.model_output_answer_cell_type
        answer_output_hidden_activation = self.hyperparams.model_output_answer_hidden_activation
        answer_output_dropout = self.hyperparams.model_output_answer_dropout if self.mode == "train" else 0.0
        answer_output_forget_bias = self.hyperparams.model_output_answer_forget_bias
        answer_output_residual_connect = self.hyperparams.model_output_answer_residual_connect
        answer_output_attention_dim = self.hyperparams.model_output_answer_attention_dim
        answer_output_score_type = self.hyperparams.model_output_answer_score_type
        answer_output_trainable = self.hyperparams.model_output_answer_trainable
        default_output_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build answer output layer")
            answer_output_list = []
            answer_output_mask_list = []
            
            with tf.variable_scope("base", reuse=tf.AUTO_REUSE):
                question_base, question_base_mask = create_base_variable(self.batch_size, question_understanding_unit_dim,
                    self.num_gpus, default_output_gpu_id, None, answer_output_trainable)
                
                answer_base_attention_layer = create_attention_layer("att",
                    question_understanding_unit_dim, question_understanding_unit_dim,
                    answer_output_attention_dim, answer_output_score_type, 0.0, False, False, False, None,
                    self.num_gpus, default_output_gpu_id, True, self.regularizer, answer_output_trainable)
                (answer_base_state, answer_base_state_mask,
                    _, _) = answer_base_attention_layer(question_base,
                        question_understanding, question_base_mask, question_understanding_mask)
            
            with tf.variable_scope("start", reuse=tf.AUTO_REUSE):                
                answer_start_attention_layer = create_attention_layer("att",
                    question_understanding_unit_dim, answer_modeling_unit_dim,
                    answer_output_attention_dim, answer_output_score_type, 0.0, False, False, False, None,
                    self.num_gpus, default_output_gpu_id, True, self.regularizer, answer_output_trainable)
                (answer_start_attention, answer_start_attention_mask, answer_start_output,
                    answer_start_output_mask) = answer_start_attention_layer(answer_base_state,
                        answer_modeling, answer_base_state_mask, answer_modeling_mask)
                answer_output_list.append(answer_start_output)
                answer_output_mask_list.append(answer_start_output_mask)
                
                answer_start_sequence_layer = create_recurrent_layer("uni", answer_output_num_layer,
                    answer_output_unit_dim, answer_output_cell_type, answer_output_hidden_activation,
                    answer_output_dropout, answer_output_forget_bias, answer_output_residual_connect,
                    None, self.num_gpus, default_output_gpu_id, True, answer_output_trainable)
                (answer_start_state,
                    answer_start_state_mask) = answer_start_sequence_layer(answer_start_attention,
                        answer_start_attention_mask)
            
            with tf.variable_scope("end", reuse=tf.AUTO_REUSE):
                answer_end_attention_layer = create_attention_layer("att",
                    answer_output_unit_dim, answer_modeling_unit_dim,
                    answer_output_attention_dim, answer_output_score_type, 0.0, False, False, False, None,
                    self.num_gpus, default_output_gpu_id, True, self.regularizer, answer_output_trainable)
                (answer_end_attention, answer_end_attention_mask, answer_end_output,
                    answer_end_output_mask) = answer_end_attention_layer(answer_start_state,
                        answer_modeling, answer_start_state_mask, answer_modeling_mask)
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
        """build graph for rnet model"""
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            """build representation layer for rnet model"""
            (question_feat, question_feat_mask, context_feat,
                context_feat_mask) = self._build_representation_layer(question_word, question_word_mask,
                    question_subword, question_subword_mask, question_char, question_char_mask, context_word,
                    context_word_mask, context_subword, context_subword_mask, context_char, context_char_mask)
            
            """build understanding layer for rnet model"""
            (question_understanding, context_understanding, question_understanding_mask,
                context_understanding_mask) = self._build_understanding_layer(question_feat,
                    context_feat, question_feat_mask, context_feat_mask)
            
            """build interaction layer for rnet model"""
            answer_interaction, answer_interaction_mask = self._build_interaction_layer(question_understanding,
                context_understanding, question_understanding_mask, context_understanding_mask)
            
            """build modeling layer for rnet model"""
            answer_modeling, answer_modeling_mask = self._build_modeling_layer(answer_interaction, answer_interaction_mask)
            
            """build output layer for rnet model"""
            (answer_output_list,
                answer_output_mask_list) = self._build_output_layer(question_understanding,
                    question_understanding_mask, answer_modeling, answer_modeling_mask)
            answer_start_output = answer_output_list[0]
            answer_end_output = answer_output_list[1]
            answer_start_output_mask = answer_output_mask_list[0]
            answer_end_output_mask = answer_output_mask_list[1]
            
        return answer_start_output, answer_end_output, answer_start_output_mask, answer_end_output_mask
    
    def save(self,
             sess,
             global_step):
        """save checkpoint for rnet model"""
        self.ckpt_saver.save(sess, self.ckpt_name, global_step=global_step)
    
    def restore(self,
                sess):
        """restore rnet model from checkpoint"""
        ckpt_file = tf.train.latest_checkpoint(self.ckpt_dir)
        if ckpt_file is not None:
            self.ckpt_saver.restore(sess, ckpt_file)
        else:
            raise FileNotFoundError("latest checkpoint file doesn't exist")

class WordFeat(object):
    """word-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 pretrained,
                 trainable=True,
                 scope="word_feat"):
        """initialize word-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pretrained = pretrained
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size,
                self.embed_dim, self.pretrained, 0, 0, self.trainable)
    
    def __call__(self,
                 input_word,
                 input_word_mask):
        """call word-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_word_embedding = self.embedding_layer(input_word)
            input_word_feat = tf.squeeze(input_word_embedding, axis=-2)
            input_word_feat_mask = input_word_mask
        
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
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size,
                self.embed_dim, False, 0, 0, self.trainable)
            
            self.conv_layer = create_convolution_layer("multi_2d", 1, self.embed_dim,
                self.unit_dim, 1, self.window_size, 1, "SAME", self.hidden_activation, self.dropout, None,
                False, False, self.num_gpus, self.default_gpu_id, True, self.regularizer, self.trainable)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, 0, 0)
    
    def __call__(self,
                 input_subword,
                 input_subword_mask):
        """call subword-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_subword_embedding = self.embedding_layer(input_subword)
            input_subword_embedding_mask = tf.expand_dims(input_subword_mask, axis=-1)
            
            (input_subword_conv,
                input_subword_conv_mask) = self.conv_layer(input_subword_embedding, input_subword_embedding_mask)
            
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
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size,
                self.embed_dim, False, 0, 0, self.trainable)
            
            self.conv_layer = create_convolution_layer("multi_2d", 1, self.embed_dim,
                self.unit_dim, 1, self.window_size, 1, "SAME", self.hidden_activation, self.dropout, None,
                False, False, self.num_gpus, self.default_gpu_id, True, self.regularizer, self.trainable)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, 0, 0)
    
    def __call__(self,
                 input_char,
                 input_char_mask):
        """call char-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_char_embedding = self.embedding_layer(input_char)
            input_char_embedding_mask = tf.expand_dims(input_char_mask, axis=-1)
            
            (input_char_conv,
                input_char_conv_mask) = self.conv_layer(input_char_embedding, input_char_embedding_mask)
            
            (input_char_pool,
                input_char_pool_mask) = self.pooling_layer(input_char_conv, input_char_conv_mask)
            
            input_char_feat = input_char_pool
            input_char_feat_mask = input_char_pool_mask
        
        return input_char_feat, input_char_feat_mask
