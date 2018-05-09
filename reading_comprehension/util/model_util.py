import collections

import numpy as np
import tensorflow as tf

from model.base_model import *
from util.data_util import *

__all__ = ["TrainModel", "EvalModel", "InferModel",
           "create_train_model", "create_eval_model", "create_infer_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "data_pipeline"))):
    pass

class EvalModel(collections.namedtuple("EvalModel", ("graph", "model", "data_pipeline"))):
    pass

class InferModel(collections.namedtuple("InferModel", ("graph", "model", "data_pipeline"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_question_data, input_context_data, input_answer_data,
             word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
             subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
             char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_mrc_data(logger,
             hyperparams.data_train_question_file, hyperparams.data_train_context_file,
             hyperparams.data_train_answer_file, hyperparams.data_answer_type, hyperparams.data_word_vocab_file, 
             hyperparams.data_word_vocab_size, hyperparams.model_representation_word_embed_dim,
             hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk,
             hyperparams.data_word_pad, hyperparams.data_word_sos, hyperparams.data_word_eos,
             hyperparams.model_representation_word_feat_enable, hyperparams.model_representation_word_embed_pretrained,
             hyperparams.data_subword_vocab_file, hyperparams.data_subword_vocab_size, hyperparams.data_subword_unk,
             hyperparams.data_subword_pad, hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable, 
             hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_unk,
             hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create question dataset")
        (input_question_word_dataset, input_question_subword_dataset,
             input_question_char_dataset) = create_src_dataset(hyperparams.data_train_question_file,
             word_vocab_index, hyperparams.data_max_question_length, hyperparams.data_word_pad,
             hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.model_representation_word_feat_enable,
             subword_vocab_index, hyperparams.data_max_subword_length, hyperparams.data_subword_pad,
             hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable, char_vocab_index,
             hyperparams.data_max_char_length, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create context dataset")
        (input_context_word_dataset, input_context_subword_dataset,
             input_context_char_dataset) = create_src_dataset(hyperparams.data_train_context_file,
             word_vocab_index, hyperparams.data_max_context_length, hyperparams.data_word_pad,
             hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.model_representation_word_feat_enable,
             subword_vocab_index, hyperparams.data_max_subword_length, hyperparams.data_subword_pad,
             hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable, char_vocab_index,
             hyperparams.data_max_char_length, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create answer dataset")
        input_answer_dataset = create_trg_dataset(hyperparams.data_train_answer_file,
            hyperparams.data_answer_type, word_vocab_index, hyperparams.data_max_answer_length,
            hyperparams.data_word_sos, hyperparams.data_word_eos)
        
        logger.log_print("# create train data pipeline")
        data_pipeline = create_data_pipeline(input_question_word_dataset,
            input_question_subword_dataset, input_question_char_dataset, input_context_word_dataset,
            input_context_subword_dataset, input_context_char_dataset, input_answer_dataset,
            word_vocab_index, hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable,
            subword_vocab_index, hyperparams.data_subword_pad, hyperparams.model_representation_subword_feat_enable,
            char_vocab_index, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable, len(input_answer_data),
            hyperparams.train_batch_size, hyperparams.train_random_seed, hyperparams.train_enable_shuffle)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline)

def create_eval_model(logger,
                      hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare evaluation data")
        logger.log_print("# create evaluation data pipeline")
        return EvalModel(graph=graph, model=model, data_pipeline=data_pipeline)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare inference data")
        logger.log_print("# create inference data pipeline")
        return InferModel(graph=graph, model=model, data_pipeline=data_pipeline)

def get_model_creator(model_type):
    if model_type == "bidaf":
        model_creator = BaseModel
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model):
    with model.graph.as_default():
        model.model.restore(sess)
