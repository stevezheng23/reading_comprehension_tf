import collections

import numpy as np
import tensorflow as tf

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
        logger.log_print("# create train data pipeline")
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

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model):
    with model.graph.as_default():
        model.model.restore(sess)
