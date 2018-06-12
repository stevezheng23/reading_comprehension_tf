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
        
        self.batch_size = tf.size(tf.reduce_max(self.data_pipeline.input_answer_mask, axis=-2))
        self.num_gpus = self.hyperparams.device_num_gpus
        self.default_gpu_id = self.hyperparams.device_default_gpu_id
        self.device_spec = get_device_spec(self.default_gpu_id, self.num_gpus)
        self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
            .format(self.num_gpus, self.default_gpu_id))
        
        self.word_embedding_layer = None
        self.word_embedding_placeholder = None
        self.subword_embedding_layer = None
        self.subword_conv_layer = None
        self.subword_pooling_layer = None
        self.char_embedding_layer = None
        self.char_conv_layer = None
        self.char_pooling_layer = None
        self.feat_fusion_layer = None
    
    def _create_fusion_layer(self,
                             input_unit_dim,
                             output_unit_dim,
                             fusion_type,
                             fusion_num_layer,
                             fusion_hidden_activation,
                             fusion_dropout,
                             fusion_trainable):
        """create fusion layer for mrc base model"""
        with tf.variable_scope("fusion", reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if fusion_type == "concate":
                fusion_layer_list = []
            elif fusion_type == "dense":
                fusion_layer = create_dense_layer(fusion_num_layer, output_unit_dim, fusion_hidden_activation,
                    fusion_dropout, self.num_gpus, self.default_gpu_id, fusion_trainable)
                fusion_layer_list = [fusion_layer]
            elif fusion_type == "highway":
                fusion_layer_list = []
                if input_unit_dim != output_unit_dim:
                    convert_layer = create_dense_layer(1, output_unit_dim, "", 0.0,
                        self.num_gpus, self.default_gpu_id, fusion_trainable)
                    fusion_layer_list.append(convert_layer)
                
                fusion_layer = create_highway_layer(fusion_num_layer, output_unit_dim, fusion_hidden_activation,
                    fusion_dropout, self.num_gpus, self.default_gpu_id, fusion_trainable)
                fusion_layer_list.append(fusion_layer)
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
    
    def _build_word_feat(self,
                         input_word,
                         input_word_mask,
                         word_vocab_size,
                         word_embed_dim,
                         word_embed_pretrained,
                         word_feat_trainable):
        """build word-level featurization for mrc base model"""
        with tf.variable_scope("feat/word", reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if self.word_embedding_layer == None:
                self.word_embedding_layer = create_embedding_layer(word_vocab_size,
                    word_embed_dim, word_embed_pretrained, 0, 0, word_feat_trainable)
            
            if self.word_embedding_placeholder == None:
                self.word_embedding_placeholder = self.word_embedding_layer.get_embedding_placeholder()
            
            input_word_embedding = self.word_embedding_layer(input_word)
            input_word_feat = tf.squeeze(input_word_embedding, axis=-2)
            input_word_feat_mask = input_word_mask
            input_word_feat = input_word_feat * input_word_feat_mask
        
        return input_word_feat, input_word_feat_mask
    
    def _build_subword_feat(self,
                            input_subword,
                            input_subword_mask,
                            subword_vocab_size,
                            subword_embed_dim,
                            subword_unit_dim,
                            subword_feat_trainable,
                            subword_max_length,
                            subword_window_size,
                            subword_hidden_activation,
                            subword_dropout,
                            subword_pooling_type):
        """build subword-level featurization for mrc base model"""
        with tf.variable_scope("feat/subword", reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if self.subword_embedding_layer == None:
                self.subword_embedding_layer = create_embedding_layer(subword_vocab_size,
                    subword_embed_dim, False, 0, 0, subword_feat_trainable)
            
            input_subword_embedding = self.subword_embedding_layer(input_subword)
            input_subword_embedding_mask = tf.expand_dims(input_subword_mask, axis=-1)
            
            if self.subword_conv_layer == None:
                self.subword_conv_layer = create_convolution_layer("multi_2d", subword_embed_dim,
                    subword_unit_dim, subword_window_size, 1, "SAME", subword_hidden_activation, 
                    subword_dropout, self.num_gpus, self.default_gpu_id, subword_feat_trainable)
            
            (input_subword_conv,
                input_subword_conv_mask) = self.subword_conv_layer(input_subword_embedding, input_subword_embedding_mask)
            
            if self.subword_pooling_layer == None:
                subword_pooling_layer = create_pooling_layer(subword_pooling_type, 0, 0)
            
            (input_subword_pool,
                input_subword_pool_mask) = self.subword_pooling_layer(input_subword_conv, input_subword_conv_mask)
            input_subword_feat = input_subword_pool
            input_subword_feat_mask = input_subword_pool_mask
        
        return input_subword_feat, input_subword_feat_mask
    
    def _build_char_feat(self,
                         input_char,
                         input_char_mask,
                         char_vocab_size,
                         char_embed_dim,
                         char_unit_dim,
                         char_feat_trainable,
                         char_max_length,
                         char_window_size,
                         char_hidden_activation,
                         char_dropout,
                         char_pooling_type):
        """build char-level featurization for mrc base model"""
        with tf.variable_scope("feat/char", reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            if self.char_embedding_layer == None:
                self.char_embedding_layer = create_embedding_layer(char_vocab_size,
                    char_embed_dim, False, 0, 0, char_feat_trainable)
            
            input_char_embedding = self.char_embedding_layer(input_char)
            input_char_embedding_mask = tf.expand_dims(input_char_mask, axis=-1)
            
            if self.char_conv_layer == None:
                self.char_conv_layer = create_convolution_layer("multi_2d", char_embed_dim,
                    char_unit_dim, char_window_size, 1, "SAME", char_hidden_activation,
                    char_dropout, self.num_gpus, self.default_gpu_id, char_feat_trainable)
            
            (input_char_conv,
                input_char_conv_mask) = self.char_conv_layer(input_char_embedding, input_char_embedding_mask)
            
            if self.char_pooling_layer == None:
                self.char_pooling_layer = create_pooling_layer(char_pooling_type, 0, 0)
            
            (input_char_pool,
                input_char_pool_mask) = self.char_pooling_layer(input_char_conv, input_char_conv_mask)
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
        subword_max_length = self.hyperparams.data_max_subword_length
        subword_window_size = self.hyperparams.model_representation_subword_window_size
        subword_hidden_activation = self.hyperparams.model_representation_subword_hidden_activation
        subword_dropout = self.hyperparams.model_representation_subword_dropout if self.mode == "train" else 0.0
        subword_pooling_type = self.hyperparams.model_representation_subword_pooling_type
        subword_feat_enable = self.hyperparams.model_representation_subword_feat_enable
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_unit_dim = self.hyperparams.model_representation_char_unit_dim
        char_feat_trainable = self.hyperparams.model_representation_char_feat_trainable
        char_max_length = self.hyperparams.data_max_char_length
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
        
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_feat_list = []
            input_feat_mask_list = []
            if word_feat_enable == True:
                input_word_feat, input_word_feat_mask = self._build_word_feat(input_word, input_word_mask,
                    word_vocab_size, word_embed_dim, word_embed_pretrained, word_feat_trainable)
                input_feat_list.append(input_word_feat)
                input_feat_mask_list.append(input_word_feat_mask)
                word_unit_dim = word_embed_dim
            else:
                word_unit_dim = 0
                self.word_embedding_placeholder = None
            
            if subword_feat_enable == True:
                input_subword_feat, input_subword_feat_mask = self._build_subword_feat(input_subword, input_subword_mask,
                    subword_vocab_size, subword_embed_dim, subword_unit_dim, subword_feat_trainable, subword_max_length,
                    subword_window_size, subword_hidden_activation, subword_dropout, subword_pooling_type)
                input_feat_list.append(input_subword_feat)
                input_feat_mask_list.append(input_subword_feat_mask)
            else:
                subword_unit_dim = 0
            
            if char_feat_enable == True:
                input_char_feat, input_char_feat_mask = self._build_char_feat(input_char, input_char_mask,
                    char_vocab_size, char_embed_dim, char_unit_dim, char_feat_trainable, char_max_length,
                    char_window_size, char_hidden_activation, char_dropout, char_pooling_type)
                input_feat_list.append(input_char_feat)
                input_feat_mask_list.append(input_char_feat_mask)
            else:
                char_unit_dim = 0
            
            feat_unit_dim = word_unit_dim + subword_unit_dim + char_unit_dim
            if self.feat_fusion_layer == None:
                self.feat_fusion_layer = self._create_fusion_layer(feat_unit_dim, fusion_unit_dim,
                    fusion_type, fusion_num_layer, fusion_hidden_activation, fusion_dropout, fusion_trainable)
            
            input_feat, input_feat_mask = self._build_fusion_result(input_feat_list,
                input_feat_mask_list, self.feat_fusion_layer)
            input_feat = input_feat * input_feat_mask
        
        return input_feat, input_feat_mask
    
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
        grads_and_vars = self.optimizer.compute_gradients(loss)
        
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

