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
        
        self.update_op = None
        self.train_loss = None
        self.learning_rate = None
        self.global_step = None
        self.train_summary = None
        self.infer_answer_start = None
        self.infer_answer_start_mask = None
        self.infer_answer_end = None
        self.infer_answer_end_mask = None
        self.infer_summary = None
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
        
        self.random_seed = self.hyperparams.train_random_seed if self.hyperparams.train_enable_debugging else None
    
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
                             random_seed,
                             trainable):
        """create fusion layer for mrc base model"""
        with tf.variable_scope("fusion", reuse=tf.AUTO_REUSE):
            if fusion_type == "concate":
                fusion_layer_list = []
                if input_unit_dim != output_unit_dim:
                    convert_layer = create_convolution_layer("1d", 1, input_unit_dim,
                        output_unit_dim, 1, 1, 1, "SAME", None, [0.0], None, False, False, False,
                        num_gpus, default_gpu_id, regularizer, random_seed, trainable)
                    fusion_layer_list.append(convert_layer)
            elif fusion_type == "dense":
                fusion_layer = create_dense_layer("single", num_layer, output_unit_dim, 1, hidden_activation,
                    [dropout] * num_layer, None, False, False, False, num_gpus, default_gpu_id, regularizer, random_seed, trainable)
                fusion_layer_list = [fusion_layer]
            elif fusion_type == "highway":
                fusion_layer_list = []
                if input_unit_dim != output_unit_dim:
                    convert_layer = create_convolution_layer("1d", 1, input_unit_dim,
                        output_unit_dim, 1, 1, 1, "SAME", None, [0.0], None, False, False, False,
                        num_gpus, default_gpu_id, regularizer, random_seed, trainable)
                    fusion_layer_list.append(convert_layer)
                
                fusion_layer = create_highway_layer(num_layer, output_unit_dim, hidden_activation,
                    [dropout] * num_layer, num_gpus, default_gpu_id, regularizer, random_seed, trainable)
                fusion_layer_list.append(fusion_layer)
            elif fusion_type == "conv":
                fusion_layer = create_convolution_layer("1d", num_layer, input_unit_dim,
                    output_unit_dim, 1, 1, 1, "SAME", hidden_activation, [dropout] * num_layer,
                    None, False, False, False, num_gpus, default_gpu_id, regularizer, random_seed, trainable)
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
        input_fusion_mask = tf.reduce_max(tf.concat(input_mask_list, axis=-1), axis=-1, keepdims=True)
        
        if fusion_layer_list != None:
            for fusion_layer in fusion_layer_list:
                input_fusion, input_fusion_mask = fusion_layer(input_fusion, input_fusion_mask)
        
        return input_fusion, input_fusion_mask
    
    def _get_exponential_moving_average(self,
                                        num_steps):
        decay_rate = self.hyperparams.train_ema_decay_rate
        enable_debias = self.hyperparams.train_ema_enable_debias
        enable_dynamic_decay = self.hyperparams.train_ema_enable_dynamic_decay
        
        if enable_dynamic_decay == True:
            ema = tf.train.ExponentialMovingAverage(decay=decay_rate, num_updates=num_steps, zero_debias=enable_debias)
        else:
            ema = tf.train.ExponentialMovingAverage(decay=decay_rate, zero_debias=enable_debias)
        
        return ema
    
    def _apply_learning_rate_warmup(self,
                                    learning_rate):
        """apply learning rate warmup"""
        warmup_mode = self.hyperparams.train_optimizer_warmup_mode
        warmup_rate = self.hyperparams.train_optimizer_warmup_rate
        warmup_end_step = self.hyperparams.train_optimizer_warmup_end_step
        
        if warmup_mode == "exponential_warmup":
            warmup_factor = warmup_rate ** (1 - tf.to_float(self.global_step) / tf.to_float(warmup_end_step))
            warmup_learning_rate = warmup_factor * learning_rate
        elif warmup_mode == "inverse_exponential_warmup":
            warmup_factor = tf.log(tf.to_float(self.global_step + 1)) / tf.log(tf.to_float(warmup_end_step))
            warmup_learning_rate = warmup_factor * learning_rate
        else:
            raise ValueError("unsupported warm-up mode {0}".format(warmup_mode))
        
        warmup_learning_rate = tf.cond(tf.less(self.global_step, warmup_end_step),
            lambda: warmup_learning_rate, lambda: learning_rate)
        
        return warmup_learning_rate
    
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
    
    def train(self,
              sess,
              word_embedding):
        """train model"""
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
        """infer model"""
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
        
        max_context_length = self.hyperparams.data_max_context_length
        max_answer_length = self.hyperparams.data_max_answer_length
        
        predict_start = np.expand_dims(answer_start[:max_context_length], axis=-1)
        predict_start_mask = np.expand_dims(answer_start_mask[:max_context_length], axis=-1)
        predict_start_start = predict_start * predict_start_mask
        predict_end = np.expand_dims(answer_end[:max_context_length], axis=-1)
        predict_end_mask = np.expand_dims(answer_end_mask[:max_context_length], axis=-1)
        predict_end = predict_end * predict_end_mask
        
        predict_span = np.matmul(predict_start, predict_end.transpose((0,2,1)))
        predict_span_mask = np.matmul(predict_start_mask, predict_end_mask.transpose((0,2,1)))
        predict_span = predict_span * predict_span_mask
        
        predict = np.full((batch_size, 2), -1)
        for k in range(batch_size):
            max_prob = float('-inf')
            max_prob_start = -1
            max_prob_end = -1
            for i in range(max_context_length):
                for j in range(i, min(max_context_length, i+max_answer_length)):
                    if predict_span[k, i, j] > max_prob:
                        max_prob = predict_span[k, i, j]
                        max_prob_start = i
                        max_prob_end = j
            
            predict[k, 0] = max_prob_start
            predict[k, 1] = max_prob_end
        
        predict_detail = np.concatenate((predict_start, predict_end), axis=-1)
        
        return InferResult(predict=predict, predict_detail=predict_detail, batch_size=batch_size, summary=summary)
    
    def _get_train_summary(self):
        """get train summary"""
        return tf.summary.merge([tf.summary.scalar("learning_rate", self.learning_rate),
            tf.summary.scalar("train_loss", self.train_loss), tf.summary.scalar("gradient_norm", self.gradient_norm)])
    
    def _get_infer_summary(self):
        """get infer summary"""
        return tf.no_op()
