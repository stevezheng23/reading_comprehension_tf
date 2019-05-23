import collections
import os.path

import numpy as np
import tensorflow as tf

from model.bidaf import *
from model.qanet import *
from model.rnet import *
from util.data_util import *

__all__ = ["TrainModel", "InferModel",
           "create_train_model", "create_infer_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data",
     "input_question", "input_question_word", "input_question_subword", "input_question_char",
     "input_context", "input_context_word", "input_context_subword", "input_context_char", "input_answer"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data",
     "input_question", "input_question_word", "input_question_subword", "input_question_char",
     "input_context", "input_context_word", "input_context_subword", "input_context_char", "input_answer"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, input_question_data, input_context_data, input_answer_data,
             word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
             subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
             char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_mrc_data(logger,
             hyperparams.data_train_mrc_file, hyperparams.data_train_mrc_file_type, hyperparams.data_answer_type,
             hyperparams.data_expand_multiple_answer, hyperparams.data_max_question_length, hyperparams.data_max_context_length,
             hyperparams.data_max_answer_length, hyperparams.data_enable_validation, hyperparams.data_word_vocab_file,
             hyperparams.data_word_vocab_size, hyperparams.data_word_vocab_threshold, hyperparams.model_representation_word_embed_dim,
             hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk,
             hyperparams.data_word_pad, hyperparams.data_word_sos, hyperparams.data_word_eos,
             hyperparams.model_representation_word_feat_enable, hyperparams.model_representation_word_embed_pretrained,
             hyperparams.data_subword_vocab_file, hyperparams.data_subword_vocab_size, hyperparams.data_subword_vocab_threshold, 
             hyperparams.data_subword_unk, hyperparams.data_subword_pad, hyperparams.data_subword_size,                                                  hyperparams.model_representation_subword_feat_enable, hyperparams.data_char_vocab_file,
             hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold, hyperparams.data_char_unk,
             hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        external_data = {}
        if word_embed_data is not None:
            external_data["word_embedding"] = word_embed_data
        
        word_vocab_tensor_index = (tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(list(word_vocab_index.keys())),
            default_value=0) if hyperparams.model_representation_word_feat_enable else None)
        subword_vocab_tensor_index = (tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(list(subword_vocab_index.keys())),
            default_value=0) if hyperparams.model_representation_subword_feat_enable else None)
        char_vocab_tensor_index = (tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(list(char_vocab_index.keys())),
            default_value=0) if hyperparams.model_representation_char_feat_enable else None)
        
        if hyperparams.data_pipeline_mode == "tfrecord":
            if not os.path.exists(hyperparams.data_tfrecord_dir):
                os.mkdir(hyperparams.data_tfrecord_dir)
            
            train_tfrecord_file = os.path.join(hyperparams.data_tfrecord_dir, "train.tfrecord")
            
            input_question_word_data = None
            input_question_subword_data = None
            input_question_char_data = None
            input_context_word_data = None
            input_context_subword_data = None
            input_context_char_data = None
            if not os.path.exists(train_tfrecord_file):
                logger.log_print("# create train question data")
                (input_question_word_data, input_question_subword_data,
                     input_question_char_data) = create_src_data(input_question_data,
                     word_vocab_index, hyperparams.data_max_question_length, hyperparams.data_word_pad,
                     hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                     hyperparams.model_representation_word_feat_enable, subword_vocab_index,
                     hyperparams.data_max_subword_length, hyperparams.data_subword_pad, hyperparams.data_subword_size,
                     hyperparams.model_representation_subword_feat_enable, char_vocab_index, hyperparams.data_max_char_length,
                     hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)

                logger.log_print("# create train context data")
                (input_context_word_data, input_context_subword_data,
                     input_context_char_data) = create_src_data(input_context_data,
                     word_vocab_index, hyperparams.data_max_context_length, hyperparams.data_word_pad,
                     hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                     hyperparams.model_representation_word_feat_enable, subword_vocab_index, 
                     hyperparams.data_max_subword_length, hyperparams.data_subword_pad, hyperparams.data_subword_size,
                     hyperparams.model_representation_subword_feat_enable, char_vocab_index, hyperparams.data_max_char_length,
                     hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)

                logger.log_print("# create train answer data")
                input_answer_data = create_trg_data(input_answer_data, hyperparams.data_answer_type,
                    word_vocab_index, hyperparams.data_max_answer_length, hyperparams.data_word_pad,
                    hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable)

                logger.log_print("# create train tfrecord file")
                create_tfrecord_file(train_tfrecord_file, input_question_word_data,
                    input_question_subword_data, input_question_char_data, input_context_word_data, input_context_subword_data,
                    input_context_char_data, input_answer_data, hyperparams.model_representation_word_feat_enable,
                    hyperparams.model_representation_subword_feat_enable, hyperparams.model_representation_char_feat_enable)
            
            logger.log_print("# generate train dataset from tfrecord")
            (input_question_word_dataset, input_question_subword_dataset, input_question_char_dataset,
                input_context_word_dataset, input_context_subword_dataset, input_context_char_dataset,
                input_answer_dataset) = generate_dataset_from_tfrecord(train_tfrecord_file,
                    hyperparams.model_representation_word_feat_enable, hyperparams.model_representation_subword_feat_enable,
                    hyperparams.model_representation_char_feat_enable, hyperparams.data_max_question_length,
                    hyperparams.data_max_context_length, hyperparams.data_max_answer_length, hyperparams.data_max_subword_length,
                    hyperparams.data_max_char_length, hyperparams.data_answer_type, hyperparams.data_num_parallel)
            
            input_question_placeholder = None
            input_question_word_placeholder = None
            input_question_subword_placeholder = None
            input_question_char_placeholder = None
            input_context_placeholder = None
            input_context_word_placeholder = None
            input_context_subword_placeholder = None
            input_context_char_placeholder = None
            input_answer_placeholder = None
        elif hyperparams.data_pipeline_mode == "preprocessing":
            logger.log_print("# create train question dataset")
            (input_question_word_data, input_question_subword_data,
                 input_question_char_data) = create_src_data(input_question_data,
                 word_vocab_index, hyperparams.data_max_question_length, hyperparams.data_word_pad, hyperparams.data_word_sos,
                 hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable, hyperparams.model_representation_word_feat_enable,
                 subword_vocab_index, hyperparams.data_max_subword_length, hyperparams.data_subword_pad,
                 hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable, char_vocab_index,
                 hyperparams.data_max_char_length, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
            
            input_question_placeholder = None
            input_question_word_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_question_length, 1], dtype=tf.int32)
                if hyperparams.model_representation_word_feat_enable else None)
            input_question_subword_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_question_length, hyperparams.data_max_subword_length], dtype=tf.int32)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_question_char_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_question_length, hyperparams.data_max_char_length], dtype=tf.int32)
                if hyperparams.model_representation_char_feat_enable else None)
            input_question_word_dataset = (tf.data.Dataset.from_tensor_slices(input_question_word_placeholder)
                if hyperparams.model_representation_word_feat_enable else None)
            input_question_subword_dataset = (tf.data.Dataset.from_tensor_slices(input_question_subword_placeholder)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_question_char_dataset = (tf.data.Dataset.from_tensor_slices(input_question_char_placeholder)
                if hyperparams.model_representation_char_feat_enable else None)
            
            logger.log_print("# create train context dataset")
            (input_context_word_data, input_context_subword_data,
                 input_context_char_data) = create_src_data(input_context_data,
                 word_vocab_index, hyperparams.data_max_context_length, hyperparams.data_word_pad, hyperparams.data_word_sos,
                 hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable, hyperparams.model_representation_word_feat_enable,
                 subword_vocab_index, hyperparams.data_max_subword_length, hyperparams.data_subword_pad,
                 hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable, char_vocab_index,
                 hyperparams.data_max_char_length, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
            
            input_context_placeholder = None
            input_context_word_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_context_length, 1], dtype=tf.int32)
                if hyperparams.model_representation_word_feat_enable else None)
            input_context_subword_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_context_length, hyperparams.data_max_subword_length], dtype=tf.int32)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_context_char_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_context_length, hyperparams.data_max_char_length], dtype=tf.int32)
                if hyperparams.model_representation_char_feat_enable else None)
            input_context_word_dataset = (tf.data.Dataset.from_tensor_slices(input_context_word_placeholder)
                if hyperparams.model_representation_word_feat_enable else None)
            input_context_subword_dataset = (tf.data.Dataset.from_tensor_slices(input_context_subword_placeholder)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_context_char_dataset = (tf.data.Dataset.from_tensor_slices(input_context_char_placeholder)
                if hyperparams.model_representation_char_feat_enable else None)
            
            logger.log_print("# create train answer dataset")
            input_answer_data = create_trg_data(input_answer_data, hyperparams.data_answer_type,
                word_vocab_index, hyperparams.data_max_answer_length, hyperparams.data_word_pad,
                hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable)
            
            input_answer_placeholder = None
            if hyperparams.data_answer_type == "span":
                input_answer_placeholder = tf.placeholder(shape=[None, 2, 1], dtype=tf.int32)
            elif hyperparams.data_answer_type == "text":
                input_answer_placeholder = tf.placeholder(shape=[None, hyperparams.data_max_answer_length, 1], dtype=tf.int32)
            
            input_answer_dataset = (tf.data.Dataset.from_tensor_slices(input_answer_placeholder)
                if input_answer_placeholder is not None else None)
        else:
            logger.log_print("# create train question dataset")
            input_question_word_data = None
            input_question_subword_data = None
            input_question_char_data = None
            input_question_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_question_word_placeholder = None
            input_question_subword_placeholder = None
            input_question_char_placeholder = None
            input_question_dataset = tf.data.Dataset.from_tensor_slices(input_question_placeholder)
            (input_question_word_dataset, input_question_subword_dataset,
                 input_question_char_dataset) = create_src_dataset(input_question_dataset,
                 word_vocab_tensor_index, hyperparams.data_max_question_length, hyperparams.data_word_pad,
                 hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                 hyperparams.model_representation_word_feat_enable, subword_vocab_tensor_index, hyperparams.data_max_subword_length,
                 hyperparams.data_subword_pad, hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable,
                 char_vocab_tensor_index, hyperparams.data_max_char_length, hyperparams.data_char_pad,
                 hyperparams.model_representation_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train context dataset")
            input_context_word_data = None
            input_context_subword_data = None
            input_context_char_data = None
            input_context_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_context_word_placeholder = None
            input_context_subword_placeholder = None
            input_context_char_placeholder = None
            input_context_dataset = tf.data.Dataset.from_tensor_slices(input_context_placeholder)
            (input_context_word_dataset, input_context_subword_dataset,
                 input_context_char_dataset) = create_src_dataset(input_context_dataset,
                 word_vocab_tensor_index, hyperparams.data_max_context_length, hyperparams.data_word_pad,
                 hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                 hyperparams.model_representation_word_feat_enable, subword_vocab_tensor_index, hyperparams.data_max_subword_length,
                 hyperparams.data_subword_pad, hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable,
                 char_vocab_tensor_index, hyperparams.data_max_char_length, hyperparams.data_char_pad,
                 hyperparams.model_representation_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train answer dataset")
            input_answer_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_answer_dataset = tf.data.Dataset.from_tensor_slices(input_answer_placeholder)
            input_answer_dataset = create_trg_dataset(input_answer_dataset,
                hyperparams.data_answer_type, word_vocab_tensor_index, hyperparams.data_max_answer_length,
                hyperparams.data_word_pad, hyperparams.data_word_sos, hyperparams.data_word_eos,
                hyperparams.data_word_placeholder_enable, hyperparams.data_num_parallel)
        
        logger.log_print("# create train data pipeline")
        data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        data_pipeline = create_data_pipeline(input_question_word_dataset,
            input_question_subword_dataset, input_question_char_dataset, input_context_word_dataset,
            input_context_subword_dataset, input_context_char_dataset, input_answer_dataset, hyperparams.data_answer_type,
            word_vocab_tensor_index, hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable,
            subword_vocab_tensor_index, hyperparams.data_subword_pad, hyperparams.model_representation_subword_feat_enable,
            char_vocab_tensor_index, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable,
            hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size, hyperparams.train_random_seed,
            input_question_placeholder, input_question_word_placeholder, input_question_subword_placeholder,
            input_question_char_placeholder, input_context_placeholder, input_context_word_placeholder,
            input_context_subword_placeholder, input_context_char_placeholder, input_answer_placeholder,
            data_size_placeholder, batch_size_placeholder)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_question=input_question_data,
            input_question_word=input_question_word_data, input_question_subword=input_question_subword_data,
            input_question_char=input_question_char_data, input_context=input_context_data,
            input_context_word=input_context_word_data, input_context_subword=input_context_subword_data,
            input_context_char=input_context_char_data, input_answer=input_answer_data)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare infer data")
        (input_data, input_question_data, input_context_data, input_answer_data,
             word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
             subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
             char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_mrc_data(logger,
             hyperparams.data_eval_mrc_file, hyperparams.data_eval_mrc_file_type, hyperparams.data_answer_type,
             hyperparams.data_expand_multiple_answer, hyperparams.data_max_question_length, hyperparams.data_max_context_length,
             hyperparams.data_max_answer_length, hyperparams.data_enable_validation, hyperparams.data_word_vocab_file,
             hyperparams.data_word_vocab_size, hyperparams.data_word_vocab_threshold, hyperparams.model_representation_word_embed_dim,
             hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk,
             hyperparams.data_word_pad, hyperparams.data_word_sos, hyperparams.data_word_eos,
             hyperparams.model_representation_word_feat_enable, hyperparams.model_representation_word_embed_pretrained,
             hyperparams.data_subword_vocab_file, hyperparams.data_subword_vocab_size, hyperparams.data_subword_vocab_threshold, 
             hyperparams.data_subword_unk, hyperparams.data_subword_pad, hyperparams.data_subword_size,                                                  hyperparams.model_representation_subword_feat_enable, hyperparams.data_char_vocab_file,
             hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold, hyperparams.data_char_unk,
             hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        external_data = {}
        if word_embed_data is not None:
            external_data["word_embedding"] = word_embed_data
        
        word_vocab_tensor_index = (tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(list(word_vocab_index.keys())),
            default_value=0) if hyperparams.model_representation_word_feat_enable else None)
        subword_vocab_tensor_index = (tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(list(subword_vocab_index.keys())),
            default_value=0) if hyperparams.model_representation_subword_feat_enable else None)
        char_vocab_tensor_index = (tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(list(char_vocab_index.keys())),
            default_value=0) if hyperparams.model_representation_char_feat_enable else None)
        
        if hyperparams.data_pipeline_mode == "tfrecord":
            if not os.path.exists(hyperparams.data_tfrecord_dir):
                os.mkdir(hyperparams.data_tfrecord_dir)

            infer_tfrecord_file = os.path.join(hyperparams.data_tfrecord_dir, "infer.tfrecord")
            
            input_question_word_data = None
            input_question_subword_data = None
            input_question_char_data = None
            input_context_word_data = None
            input_context_subword_data = None
            input_context_char_data = None
            if not os.path.exists(infer_tfrecord_file):               
                logger.log_print("# create infer question data")
                (input_question_word_data, input_question_subword_data,
                     input_question_char_data) = create_src_data(input_question_data,
                     word_vocab_index, hyperparams.data_max_question_length, hyperparams.data_word_pad,
                     hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                     hyperparams.model_representation_word_feat_enable, subword_vocab_index,
                     hyperparams.data_max_subword_length, hyperparams.data_subword_pad, hyperparams.data_subword_size,
                     hyperparams.model_representation_subword_feat_enable, char_vocab_index, hyperparams.data_max_char_length,
                     hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)

                logger.log_print("# create infer context data")
                (input_context_word_data, input_context_subword_data,
                     input_context_char_data) = create_src_data(input_context_data,
                     word_vocab_index, hyperparams.data_max_context_length, hyperparams.data_word_pad,
                     hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                     hyperparams.model_representation_word_feat_enable, subword_vocab_index, 
                     hyperparams.data_max_subword_length, hyperparams.data_subword_pad, hyperparams.data_subword_size,
                     hyperparams.model_representation_subword_feat_enable, char_vocab_index, hyperparams.data_max_char_length,
                     hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)

                logger.log_print("# create infer answer data")
                input_answer_data = create_trg_data(input_answer_data, hyperparams.data_answer_type,
                    word_vocab_index, hyperparams.data_max_answer_length, hyperparams.data_word_pad,
                    hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable)

                logger.log_print("# create infer tfrecord file")
                create_tfrecord_file(infer_tfrecord_file, input_question_word_data,
                    input_question_subword_data, input_question_char_data, input_context_word_data, input_context_subword_data,
                    input_context_char_data, input_answer_data, hyperparams.model_representation_word_feat_enable,
                    hyperparams.model_representation_subword_feat_enable, hyperparams.model_representation_char_feat_enable)
            
            logger.log_print("# generate infer dataset from tfrecord")
            (input_question_word_dataset, input_question_subword_dataset, input_question_char_dataset,
                input_context_word_dataset, input_context_subword_dataset, input_context_char_dataset,
                input_answer_dataset) = generate_dataset_from_tfrecord(infer_tfrecord_file,
                    hyperparams.model_representation_word_feat_enable, hyperparams.model_representation_subword_feat_enable,
                    hyperparams.model_representation_char_feat_enable, hyperparams.data_max_question_length,
                    hyperparams.data_max_context_length, hyperparams.data_max_answer_length, hyperparams.data_max_subword_length,
                    hyperparams.data_max_char_length, hyperparams.data_answer_type, hyperparams.data_num_parallel)
            
            input_question_placeholder = None
            input_question_word_placeholder = None
            input_question_subword_placeholder = None
            input_question_char_placeholder = None
            input_context_placeholder = None
            input_context_word_placeholder = None
            input_context_subword_placeholder = None
            input_context_char_placeholder = None
            input_answer_placeholder = None
        elif hyperparams.data_pipeline_mode == "preprocessing":
            logger.log_print("# create infer question dataset")
            (input_question_word_data, input_question_subword_data,
                 input_question_char_data) = create_src_data(input_question_data,
                 word_vocab_index, hyperparams.data_max_question_length, hyperparams.data_word_pad, hyperparams.data_word_sos,
                 hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable, hyperparams.model_representation_word_feat_enable,
                 subword_vocab_index, hyperparams.data_max_subword_length, hyperparams.data_subword_pad,
                 hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable, char_vocab_index,
                 hyperparams.data_max_char_length, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
            
            input_question_placeholder = None
            input_question_word_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_question_length, 1], dtype=tf.int32)
                if hyperparams.model_representation_word_feat_enable else None)
            input_question_subword_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_question_length, hyperparams.data_max_subword_length], dtype=tf.int32)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_question_char_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_question_length, hyperparams.data_max_char_length], dtype=tf.int32)
                if hyperparams.model_representation_char_feat_enable else None)
            input_question_word_dataset = (tf.data.Dataset.from_tensor_slices(input_question_word_placeholder)
                if hyperparams.model_representation_word_feat_enable else None)
            input_question_subword_dataset = (tf.data.Dataset.from_tensor_slices(input_question_subword_placeholder)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_question_char_dataset = (tf.data.Dataset.from_tensor_slices(input_question_char_placeholder)
                if hyperparams.model_representation_char_feat_enable else None)
            
            logger.log_print("# create infer context dataset")
            (input_context_word_data, input_context_subword_data,
                 input_context_char_data) = create_src_data(input_context_data,
                 word_vocab_index, hyperparams.data_max_context_length, hyperparams.data_word_pad, hyperparams.data_word_sos,
                 hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable, hyperparams.model_representation_word_feat_enable,
                 subword_vocab_index, hyperparams.data_max_subword_length, hyperparams.data_subword_pad,
                 hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable, char_vocab_index,
                 hyperparams.data_max_char_length, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
            
            input_context_placeholder = None
            input_context_word_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_context_length, 1], dtype=tf.int32)
                if hyperparams.model_representation_word_feat_enable else None)
            input_context_subword_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_context_length, hyperparams.data_max_subword_length], dtype=tf.int32)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_context_char_placeholder = (tf.placeholder(
                shape=[None, hyperparams.data_max_context_length, hyperparams.data_max_char_length], dtype=tf.int32)
                if hyperparams.model_representation_char_feat_enable else None)
            input_context_word_dataset = (tf.data.Dataset.from_tensor_slices(input_context_word_placeholder)
                if hyperparams.model_representation_word_feat_enable else None)
            input_context_subword_dataset = (tf.data.Dataset.from_tensor_slices(input_context_subword_placeholder)
                if hyperparams.model_representation_subword_feat_enable else None)
            input_context_char_dataset = (tf.data.Dataset.from_tensor_slices(input_context_char_placeholder)
                if hyperparams.model_representation_char_feat_enable else None)
            
            logger.log_print("# create infer answer dataset")
            input_answer_data = create_trg_data(input_answer_data, hyperparams.data_answer_type,
                word_vocab_index, hyperparams.data_max_answer_length, hyperparams.data_word_pad,
                hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable)
            
            input_answer_placeholder = None
            if hyperparams.data_answer_type == "span":
                input_answer_placeholder = tf.placeholder(shape=[None, 2, 1], dtype=tf.int32)
            elif hyperparams.data_answer_type == "text":
                input_answer_placeholder = tf.placeholder(shape=[None, hyperparams.data_max_answer_length, 1], dtype=tf.int32)
            
            input_answer_dataset = (tf.data.Dataset.from_tensor_slices(input_answer_placeholder)
                if input_answer_placeholder is not None else None)
        else:
            logger.log_print("# create infer question dataset")
            input_question_word_data = None
            input_question_subword_data = None
            input_question_char_data = None
            input_question_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_question_word_placeholder = None
            input_question_subword_placeholder = None
            input_question_char_placeholder = None
            input_question_dataset = tf.data.Dataset.from_tensor_slices(input_question_placeholder)
            (input_question_word_dataset, input_question_subword_dataset,
                 input_question_char_dataset) = create_src_dataset(input_question_dataset,
                 word_vocab_tensor_index, hyperparams.data_max_question_length, hyperparams.data_word_pad,
                 hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                 hyperparams.model_representation_word_feat_enable, subword_vocab_tensor_index, hyperparams.data_max_subword_length,
                 hyperparams.data_subword_pad, hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable,
                 char_vocab_tensor_index, hyperparams.data_max_char_length, hyperparams.data_char_pad,
                 hyperparams.model_representation_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create infer context dataset")
            input_context_word_data = None
            input_context_subword_data = None
            input_context_char_data = None
            input_context_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_context_word_placeholder = None
            input_context_subword_placeholder = None
            input_context_char_placeholder = None
            input_context_dataset = tf.data.Dataset.from_tensor_slices(input_context_placeholder)
            (input_context_word_dataset, input_context_subword_dataset,
                 input_context_char_dataset) = create_src_dataset(input_context_dataset,
                 word_vocab_tensor_index, hyperparams.data_max_context_length, hyperparams.data_word_pad,
                 hyperparams.data_word_sos, hyperparams.data_word_eos, hyperparams.data_word_placeholder_enable,
                 hyperparams.model_representation_word_feat_enable, subword_vocab_tensor_index, hyperparams.data_max_subword_length,
                 hyperparams.data_subword_pad, hyperparams.data_subword_size, hyperparams.model_representation_subword_feat_enable,
                 char_vocab_tensor_index, hyperparams.data_max_char_length, hyperparams.data_char_pad,
                 hyperparams.model_representation_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create infer answer dataset")
            input_answer_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_answer_dataset = tf.data.Dataset.from_tensor_slices(input_answer_placeholder)
            input_answer_dataset = create_trg_dataset(input_answer_dataset,
                hyperparams.data_answer_type, word_vocab_tensor_index, hyperparams.data_max_answer_length,
                hyperparams.data_word_pad, hyperparams.data_word_sos, hyperparams.data_word_eos,
                hyperparams.data_word_placeholder_enable, hyperparams.data_num_parallel)
        
        logger.log_print("# create infer data pipeline")
        data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        data_pipeline = create_data_pipeline(input_question_word_dataset,
            input_question_subword_dataset, input_question_char_dataset, input_context_word_dataset,
            input_context_subword_dataset, input_context_char_dataset, input_answer_dataset, hyperparams.data_answer_type,
            word_vocab_tensor_index, hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable,
            subword_vocab_tensor_index, hyperparams.data_subword_pad, hyperparams.model_representation_subword_feat_enable,
            char_vocab_tensor_index, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable, False, 0, 0,
            input_question_placeholder, input_question_word_placeholder, input_question_subword_placeholder,
            input_question_char_placeholder, input_context_placeholder, input_context_word_placeholder,
            input_context_subword_placeholder, input_context_char_placeholder, input_answer_placeholder,
            data_size_placeholder, batch_size_placeholder)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="infer", scope=hyperparams.model_scope)
        
        return InferModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_question=input_question_data,
            input_question_word=input_question_word_data, input_question_subword=input_question_subword_data,
            input_question_char=input_question_char_data, input_context=input_context_data,
            input_context_word=input_context_word_data, input_context_subword=input_context_subword_data,
            input_context_char=input_context_char_data, input_answer=input_answer_data)

def get_model_creator(model_type):
    if model_type == "bidaf":
        model_creator = BiDAF
    elif model_type == "qanet":
        model_creator = QANet
    elif model_type == "rnet":
        model_creator = RNet
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model,
               ckpt_file,
               ckpt_type):
    with model.graph.as_default():
        model.model.restore(sess, ckpt_file, ckpt_type)
