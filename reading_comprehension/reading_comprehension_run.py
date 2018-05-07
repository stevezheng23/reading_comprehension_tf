import argparse
import json

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.param_util import *
from util.model_util import *
from util.debug_logger import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def train(logger,
          hyperparams):
    logger.log_print("##### create train model #####")
    train_model = create_train_model(logger, hyperparams)
    logger.log_print("##### create eval model #####")
    eval_model = create_eval_model(logger, hyperparams)
    logger.log_print("##### create infer model #####")
    infer_model = create_infer_model(logger, hyperparams)
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
    
    logger.log_print("##### start training #####")
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    train_summary_writer = SummaryWriter(train_model.graph, os.path.join(summary_output_dir, "train"))
    eval_summary_writer = SummaryWriter(eval_model.graph, os.path.join(summary_output_dir, "eval"))
    infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
    
    init_model(train_sess, train_model)
    init_model(eval_sess, eval_model)
    init_model(infer_sess, infer_model)
    
    global_step = 0
    train_model.model.save(train_sess, global_step)
    train_logger = TrainLogger(hyperparams.data_log_output_dir)
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    for epoch in range(hyperparams.train_num_epoch):
        train_sess.run(train_model.data_pipeline.initializer)
        step_in_epoch = 0
        while True:
            try:
                start_time = time.time()
                train_result = train_model.model.train(train_sess,
                    train_model.src_embedding, train_model.trg_embedding)
                end_time = time.time()
                
                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)

                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                    train_summary_writer.add_summary(train_result.summary, global_step)
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step)
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_model.model.save(train_sess, global_step)
                break

    train_summary_writer.close_writer()
    eval_summary_writer.close_writer()
    infer_summary_writer.close_writer()
    logger.log_print("##### finish training #####")

def evaluate(logger,
             hyperparams):
    pass

def test(logger,
         hyperparams):   
    train_model = create_train_model(logger, hyperparams)
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    
    init_model(train_sess, train_model)
    
    train_sess.run(train_model.data_pipeline.initializer)
    
    (input_question_word, input_question_subword, input_question_char, input_question_word_feat,
         input_question_subword_feat, input_question_char_feat) = train_sess.run([train_model.data_pipeline.input_question_word,
             train_model.data_pipeline.input_question_subword, train_model.data_pipeline.input_question_char,
             train_model.model.input_question_word_feat, train_model.model.input_question_subword_feat,
             train_model.model.input_question_char_feat])
    print(input_question_word)
    print(input_question_subword)
    print(input_question_char)
    print(input_question_word_feat)
    print(input_question_subword_feat)
    print(input_question_char_feat)

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train'):
        train(logger, hyperparams)
    elif (args.mode == 'eval'):
        evaluate(logger, hyperparams)
    elif (args.mode == 'test'):
        test(logger, hyperparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
