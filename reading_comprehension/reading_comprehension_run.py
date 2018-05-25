import argparse
import os.path
import time

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.param_util import *
from util.model_util import *
from util.debug_logger import *
from util.train_logger import *
from util.eval_logger import *
from util.summary_writer import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def extrinsic_eval(logger,
                   summary_writer,
                   sess,
                   model,
                   question_data,
                   context_data,
                   answer_data,
                   word_embedding,
                   batch_size,
                   metric_list,
                   global_step):
    load_model(sess, model)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.data_pipeline.input_question_placeholder: question_data,
            model.data_pipeline.input_context_placeholder: context_data,
            model.data_pipeline.input_answer_placeholder: answer_data,
            model.data_pipeline.data_size_placeholder: len(answer_data),
            model.data_pipeline.batch_size_placeholder: batch_size})
    
    predict = []
    while True:
        try:
            infer_result = model.model.infer(sess, word_embedding, word_embedding)
            predict.extend(infer_result.predict)
        except  tf.errors.OutOfRangeError:
            break
    
    eval_result_list = []
    for metric in metric_list:
        score = evaluate_from_data(predict, answer_data, metric)
        summary_writer.add_value_summary(metric, score, global_step)
        eval_result = ExtrinsicEvalLog(metric=metric,
            score=score, sample_output=predict, sample_size=len(predict))
        eval_result_list.append(eval_result)
    
    logger.update_extrinsic_eval(eval_result_list)
    logger.check_extrinsic_eval()

def train(logger,
          hyperparams):
    logger.log_print("##### create train model #####")
    train_model = create_train_model(logger, hyperparams)
    logger.log_print("##### create infer model #####")
    infer_model = create_infer_model(logger, hyperparams)
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
    
    logger.log_print("##### start training #####")
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    train_summary_writer = SummaryWriter(train_model.graph, os.path.join(summary_output_dir, "train"))
    infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
    
    init_model(train_sess, train_model)
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
                    train_model.word_embedding, train_model.word_embedding)
                end_time = time.time()
                
                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)
                
                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                    train_summary_writer.add_summary(train_result.summary, global_step)
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step)
                if step_in_epoch % hyperparams.train_step_per_eval == 0:
                    extrinsic_eval(eval_logger, infer_summary_writer, infer_sess,
                        infer_model, infer_model.input_question, infer_model.input_context,
                        infer_model.input_answer, infer_model.word_embedding,
                        hyperparams.train_eval_batch_size, hyperparams.train_eval_metric, global_step)
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_summary_writer.add_summary(train_result.summary, global_step)
                train_model.model.save(train_sess, global_step)
                extrinsic_eval(eval_logger, infer_summary_writer, infer_sess,
                    infer_model, infer_model.input_question, infer_model.input_context,
                    infer_model.input_answer, infer_model.word_embedding,
                    hyperparams.train_eval_batch_size, hyperparams.train_eval_metric, global_step)
                break

    train_summary_writer.close_writer()
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
    
    (input_question_word, input_context_word, input_question_word_mask, input_context_word_mask,
        input_question_char, input_context_char, input_question_char_mask, input_context_char_mask,
        input_answer, input_answer_mask, answer_start_output, answer_start_output_mask, answer_end_output,
        answer_end_output_mask) = train_sess.run([train_model.data_pipeline.input_question_word, 
            train_model.data_pipeline.input_context_word, train_model.data_pipeline.input_question_word_mask, 
            train_model.data_pipeline.input_context_word_mask, train_model.data_pipeline.input_question_char,
            train_model.data_pipeline.input_context_char, train_model.data_pipeline.input_question_char_mask,
            train_model.data_pipeline.input_context_char_mask, train_model.data_pipeline.input_answer,
            train_model.data_pipeline.input_answer_mask, train_model.model.answer_start_output,
            train_model.model.answer_start_output_mask, train_model.model.answer_end_output,
            train_model.model.answer_end_output_mask])
    print(input_question_word.shape)
    print(input_context_word.shape)
    print(input_question_word_mask.shape)
    print(input_context_word_mask.shape)
    print(input_question_char.shape)
    print(input_context_char.shape)
    print(input_question_char_mask.shape)
    print(input_context_char_mask.shape)
    print(input_answer.shape)
    print(input_answer_mask.shape)
    print(answer_start_output.shape)
    print(answer_end_output.shape)
    print(answer_start_output_mask.shape)
    print(answer_end_output_mask.shape)

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
