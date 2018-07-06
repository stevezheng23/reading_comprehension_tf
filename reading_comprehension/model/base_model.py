import collections
import os.path

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.reading_comprehension_util import *
from util.layer_util import *

__all__ = ["TrainResult", "InferResult", "BaseModel"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class InferResult(collections.namedtuple("InferResult",
    ("predict", "predict_detail", "batch_size", "summary"))):
    pass

class BaseModel(object):
    """reading comprehension base model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="base"):
        """initialize mrc base model"""
        self.logger = logger
        self.hyperparams = hyperparams
        self.data_pipeline = data_pipeline
        self.mode = mode
        self.scope = scope
        
        self.word_embedding_placeholder = None
        self.batch_size = tf.size(tf.reduce_max(self.data_pipeline.input_answer_mask, axis=-2))
        
        self.num_gpus = self.hyperparams.device_num_gpus
        self.default_gpu_id = self.hyperparams.device_default_gpu_id
        self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
            .format(self.num_gpus, self.default_gpu_id))
        
        if self.hyperparams.train_regularization_enable == True:
            self.regularizer = create_weight_regularizer(self.hyperparams.train_regularization_type,
                self.hyperparams.train_regularization_scale)
        else:
            self.regularizer = None
    
    def _create_fusion_layer(self,
                             input_unit_dim,
                             output_unit_dim,
                             fusion_type,
                             num_layer,
                             hidden_activation,
                             dropout,
                             num_gpus,
                             default_gpu_id,
                             regularizer,
                             trainable):
        """create fusion layer for mrc base model"""
        with tf.variable_scope("fusion", reuse=tf.AUTO_REUSE):
            if fusion_type == "concate":
                fusion_layer_list = []
                if input_unit_dim != output_unit_dim:
                    convert_layer = create_dense_layer(1, output_unit_dim, "", 0.0, False, False,
                        num_gpus, default_gpu_id, True, trainable)
                    fusion_layer_list.append(convert_layer)
            elif fusion_type == "dense":
                fusion_layer = create_dense_layer(num_layer, output_unit_dim, hidden_activation,
                    dropout, False, False, num_gpus, default_gpu_id, True, trainable)
                fusion_layer_list = [fusion_layer]
            elif fusion_type == "highway":
                fusion_layer_list = []
                if input_unit_dim != output_unit_dim:
                    convert_layer = create_dense_layer(1, output_unit_dim, "", 0.0, False, False,
                        num_gpus, default_gpu_id, True, trainable)
                    fusion_layer_list.append(convert_layer)
                
                fusion_layer = create_highway_layer(num_layer, output_unit_dim, hidden_activation,
                    dropout, num_gpus, default_gpu_id, True, trainable)
                fusion_layer_list.append(fusion_layer)
            elif fusion_type == "conv":
                fusion_layer = create_convolution_layer("1d", num_layer, input_unit_dim, output_unit_dim, 1, 1, 1,
                    "SAME", hidden_activation, dropout, False, False, num_gpus, default_gpu_id, True, regularizer, trainable)
                fusion_layer_list = [fusion_layer]
            else:
                raise ValueError("unsupported fusion type {0}".format(fusion_type))
        
        return fusion_layer_list
    
    def _build_fusion_result(self,
                             input_data_list,
                             input_mask_list,
                             fusion_layer_list):
        """build fusion result for mrc base model"""
        input_fusion = tf.concat(input_data_list, axis=-1)
        input_fusion_mask = tf.reduce_max(tf.concat(input_mask_list, axis=-1), axis=-1, keep_dims=True)
        
        if fusion_layer_list != None:
            for fusion_layer in fusion_layer_list:
                input_fusion, input_fusion_mask = fusion_layer(input_fusion, input_fusion_mask)
        
        return input_fusion, input_fusion_mask
    
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
        """build representation layer for mrc base model"""
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
                    pretrained=word_embed_pretrained, trainable=word_feat_trainable)
                
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
                    default_gpu_id=default_representation_gpu_id, regularizer=self.regularizer, trainable=subword_feat_trainable)
                
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
                    default_gpu_id=default_representation_gpu_id, regularizer=self.regularizer, trainable=char_feat_trainable)
                
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
                self.num_gpus, default_representation_gpu_id, self.regularizer, fusion_trainable)
            
            input_question_feat, input_question_feat_mask = self._build_fusion_result(input_question_feat_list,
                input_question_feat_mask_list, feat_fusion_layer)
            input_context_feat, input_context_feat_mask = self._build_fusion_result(input_context_feat_list,
                input_context_feat_mask_list, feat_fusion_layer)
        
        return input_question_feat, input_question_feat_mask, input_context_feat, input_context_feat_mask
    
    def _apply_learning_rate_decay(self,
                                   learning_rate):
        """apply learning rate decay"""
        decay_mode = self.hyperparams.train_optimizer_decay_mode
        decay_rate = self.hyperparams.train_optimizer_decay_rate
        decay_step = self.hyperparams.train_optimizer_decay_step
        decay_start_step = self.hyperparams.train_optimizer_decay_start_step
        
        if decay_mode == "exponential_decay":
            decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step),
                decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
        elif decay_mode == "inverse_time_decay":
            decayed_learning_rate = tf.train.inverse_time_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step),
                decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
        else:
            raise ValueError("unsupported decay mode {0}".format(decay_mode))
        
        decayed_learning_rate = tf.cond(tf.less(self.global_step, decay_start_step),
            lambda: learning_rate, lambda: decayed_learning_rate)
        
        return decayed_learning_rate
    
    def _initialize_optimizer(self,
                              learning_rate):
        """initialize optimizer"""
        optimizer_type = self.hyperparams.train_optimizer_type
        if optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                momentum=self.hyperparams.train_optimizer_momentum_beta)
        elif optimizer_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                decay=self.hyperparams.train_optimizer_rmsprop_beta,
                epsilon=self.hyperparams.train_optimizer_rmsprop_epsilon)
        elif optimizer_type == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                rho=self.hyperparams.train_optimizer_adadelta_rho,
                epsilon=self.hyperparams.train_optimizer_adadelta_epsilon)
        elif optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                initial_accumulator_value=self.hyperparams.train_optimizer_adagrad_init_accumulator)
        elif optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                beta1=self.hyperparams.train_optimizer_adam_beta_1, beta2=self.hyperparams.train_optimizer_adam_beta_2,
                epsilon=self.hyperparams.train_optimizer_adam_epsilon)
        else:
            raise ValueError("unsupported optimizer type {0}".format(optimizer_type))
        
        return optimizer
    
    def _minimize_loss(self,
                       loss):
        """minimize optimization loss"""
        """compute gradients"""
        if self.num_gpus > 1:
            grads_and_vars = self.optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
        else:
            grads_and_vars = self.optimizer.compute_gradients(loss, colocate_gradients_with_ops=False)
        
        """clip gradients"""
        gradients = [x[0] for x in grads_and_vars]
        variables = [x[1] for x in grads_and_vars]
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.hyperparams.train_clip_norm)
        grads_and_vars = zip(clipped_gradients, variables)
        
        """update model based on gradients"""
        update_model = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        
        return update_model, clipped_gradients, gradient_norm
    
    def _get_train_summary(self):
        """get train summary"""
        return tf.summary.merge([tf.summary.scalar("learning_rate", self.decayed_learning_rate),
            tf.summary.scalar("train_loss", self.train_loss), tf.summary.scalar("gradient_norm", self.gradient_norm)])
    
    def _get_infer_summary(self):
        """get infer summary"""
        return tf.no_op()

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
                self.unit_dim, 1, self.window_size, 1, "SAME", self.hidden_activation, self.dropout,
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
                self.unit_dim, 1, self.window_size, 1, "SAME", self.hidden_activation, self.dropout,
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
