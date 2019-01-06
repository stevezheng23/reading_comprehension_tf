import argparse
import os.path
import time

import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug

from util.default_util import *
from util.param_util import *
from util.model_util import *
from util.eval_util import *
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
                   enable_prepro,
                   batch_size,
                   metric_list,
                   detail_type,
                   global_step,
                   epoch,
                   ckpt_file,
                   eval_mode):
    data_size = len(model.input_data)
    load_model(sess, model, ckpt_file, eval_mode)
    
    if enable_prepro == True:
        feed_dict = {model.data_pipeline.input_answer_placeholder: model.input_answer,
            model.data_pipeline.data_size_placeholder: data_size,
            model.data_pipeline.batch_size_placeholder: batch_size}
        
        if model.data_pipeline.input_question_word_placeholder is not None and model.input_question_word is not None:
            feed_dict[model.data_pipeline.input_question_word_placeholder] = model.input_question_word
        if model.data_pipeline.input_question_subword_placeholder is not None and model.input_question_subword is not None:
            feed_dict[model.data_pipeline.input_question_subword_placeholder] = model.input_question_subword
        if model.data_pipeline.input_question_char_placeholder is not None and model.input_question_char is not None:
            feed_dict[model.data_pipeline.input_question_char_placeholder] = model.input_question_char
        if model.data_pipeline.input_context_word_placeholder is not None and model.input_context_word is not None:
            feed_dict[model.data_pipeline.input_context_word_placeholder] = model.input_context_word
        if model.data_pipeline.input_context_subword_placeholder is not None and model.input_context_subword is not None:
            feed_dict[model.data_pipeline.input_context_subword_placeholder] = model.input_context_subword
        if model.data_pipeline.input_context_char_placeholder is not None and model.input_context_char is not None:
            feed_dict[model.data_pipeline.input_context_char_placeholder] = model.input_context_char
    else:
        feed_dict={model.data_pipeline.input_question_placeholder: model.input_question,
            model.data_pipeline.input_context_placeholder: model.input_context,
            model.data_pipeline.input_answer_placeholder: model.input_answer,
            model.data_pipeline.data_size_placeholder: data_size,
            model.data_pipeline.batch_size_placeholder: batch_size}
    
    sess.run(model.data_pipeline.initializer, feed_dict=feed_dict)
    
    predict_span = []
    while True:
        try:
            infer_result = model.model.infer(sess, model.word_embedding)
            predict_span.extend(infer_result.predict)
        except  tf.errors.OutOfRangeError:
            break
    
    sample_output = []
    predict_text = []
    label_text = []
    for i in range(data_size):
        sample_id = model.input_data[i]["id"]
        context = model.input_context[i].split(" ")
        
        predict_start = int(predict_span[i][0])
        predict_end = int(predict_span[i][1])
        predict = " ".join(context[predict_start:predict_end+1])
        predict_text.append(predict)
        
        label_text.append([])
        sample_output.append({
            "id": sample_id,
            "predict": {
                "text": predict,
                "start": predict_start,
                "end": predict_end
            },
            "answers": []
        })
        
        for answer in model.input_data[i]["answers"]:
            label_start = int(answer["start"])
            label_end = int(answer["end"])
            label = " ".join(context[label_start:label_end+1])
            label_text[-1].append(label)
            
            sample_output[-1]["answers"].append({
                "text": label,
                "start": label_start,
                "end": label_end
            })
    
    eval_result_list = []
    for metric in metric_list:
        score = evaluate_from_data(predict_text, label_text, metric)
        summary_writer.add_value_summary(metric, score, global_step)
        eval_result = ExtrinsicEvalLog(metric=metric,
            score=score, sample_output=None, sample_size=len(sample_output))
        eval_result_list.append(eval_result)
    
    if detail_type == "simplified":
        sample_output = { sample["id"]: sample["predict"]["text"] for sample in sample_output }
    
    eval_result_detail = ExtrinsicEvalLog(metric="detail",
        score=0.0, sample_output=sample_output, sample_size=len(sample_output))
    basic_info = BasicInfoEvalLog(epoch=epoch, global_step=global_step)
    
    logger.update_extrinsic_eval(eval_result_list, basic_info)
    logger.update_extrinsic_eval_detail(eval_result_detail, basic_info)
    logger.check_extrinsic_eval()
    logger.check_extrinsic_eval_detail()

def decoding_eval(logger,
                  summary_writer,
                  sess,
                  model,
                  enable_prepro,
                  sample_size,
                  random_seed,
                  global_step,
                  epoch,
                  ckpt_file,
                  eval_mode):
    np.random.seed(random_seed)
    sample_ids = np.random.randint(0, len(model.input_data)-1, size=sample_size)
    sample_input_data = [model.input_data[sample_id] for sample_id in sample_ids]
    sample_question_data = [model.input_question[sample_id] for sample_id in sample_ids]
    sample_context_data = [model.input_context[sample_id] for sample_id in sample_ids]
    sample_answer_data = [model.input_answer[sample_id] for sample_id in sample_ids]
    load_model(sess, model, ckpt_file, eval_mode)
    
    if enable_prepro == True:
        feed_dict={model.data_pipeline.input_answer_placeholder: sample_answer_data,
            model.data_pipeline.data_size_placeholder: sample_size,
            model.data_pipeline.batch_size_placeholder: sample_size}
        
        if model.data_pipeline.input_question_word_placeholder is not None and model.input_question_word is not None:
            sample_question_word = [model.input_question_word[sample_id] for sample_id in sample_ids]
            feed_dict[model.data_pipeline.input_question_word_placeholder] = sample_question_word
        if model.data_pipeline.input_question_subword_placeholder is not None and model.input_question_subword is not None:
            sample_question_subword = [model.input_question_subword[sample_id] for sample_id in sample_ids]
            feed_dict[model.data_pipeline.input_question_subword_placeholder] = sample_question_subword
        if model.data_pipeline.input_question_char_placeholder is not None and model.input_question_char is not None:
            sample_question_char = [model.input_question_char[sample_id] for sample_id in sample_ids]
            feed_dict[model.data_pipeline.input_question_char_placeholder] = sample_question_char
        if model.data_pipeline.input_context_word_placeholder is not None and model.input_context_word is not None:
            sample_context_word = [model.input_context_word[sample_id] for sample_id in sample_ids]
            feed_dict[model.data_pipeline.input_context_word_placeholder] = sample_context_word
        if model.data_pipeline.input_context_subword_placeholder is not None and model.input_context_subword is not None:
            sample_context_subword = [model.input_context_subword[sample_id] for sample_id in sample_ids]
            feed_dict[model.data_pipeline.input_context_subword_placeholder] = sample_context_subword
        if model.data_pipeline.input_context_char_placeholder is not None and model.input_context_char is not None:
            sample_context_char = [model.input_context_char[sample_id] for sample_id in sample_ids]
            feed_dict[model.data_pipeline.input_context_char_placeholder] = sample_context_char
    else:
        feed_dict={model.data_pipeline.input_question_placeholder: sample_question_data,
            model.data_pipeline.input_context_placeholder: sample_context_data,
            model.data_pipeline.input_answer_placeholder: sample_answer_data,
            model.data_pipeline.data_size_placeholder: sample_size,
            model.data_pipeline.batch_size_placeholder: sample_size}
    
    sess.run(model.data_pipeline.initializer, feed_dict=feed_dict)
    
    sample_output_span = []
    while True:
        try:
            infer_result = model.model.infer(sess, model.word_embedding)
            sample_output_span.extend(infer_result.predict)
        except  tf.errors.OutOfRangeError:
            break
    
    if infer_result.summary is not None:
        summary_writer.add_summary(infer_result.summary, global_step)
    
    eval_result_list = []
    for i in range(sample_size):
        sample_input = sample_input_data[i]
        sample_context = sample_context_data[i].split(" ")
        
        output_start = int(sample_output_span[i][0])
        output_end = int(sample_output_span[i][1])
        sample_output = " ".join(sample_context[output_start:output_end+1])
        
        sample_reference_list = []
        for sample_answer in sample_input_data[i]["answers"]:
            reference_start = int(sample_answer["start"])
            reference_end = int(sample_answer["end"])
            sample_reference = " ".join(sample_context[reference_start:reference_end+1])
            sample_reference_list.append(sample_reference)
        
        eval_result = DecodingEvalLog(sample_input=sample_input,
            sample_output=sample_output, sample_reference=sample_reference_list)
        eval_result_list.append(eval_result)
    
    basic_info = BasicInfoEvalLog(epoch=epoch, global_step=global_step)
    
    logger.update_decoding_eval(eval_result_list, basic_info)
    logger.check_decoding_eval()

def train(logger,
          hyperparams,
          enable_eval=True,
          enable_debug=False):
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    logger.log_print("##### create train model #####")
    train_model = create_train_model(logger, hyperparams)
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    if enable_debug == True:
        train_sess = tf_debug.LocalCLIDebugWrapperSession(train_sess)
    
    train_summary_writer = SummaryWriter(train_model.graph, os.path.join(summary_output_dir, "train"))
    init_model(train_sess, train_model)
    train_logger = TrainLogger(hyperparams.data_log_output_dir)
    
    if enable_eval == True:
        logger.log_print("##### create infer model #####")
        infer_model = create_infer_model(logger, hyperparams)
        infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
        if enable_debug == True:
            infer_sess = tf_debug.LocalCLIDebugWrapperSession(infer_sess)
        
        infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
        init_model(infer_sess, infer_model)
        eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    
    logger.log_print("##### start training #####")
    global_step = 0
    train_model.model.save(train_sess, global_step, "debug")
    for epoch in range(hyperparams.train_num_epoch):
        train_sess.run(train_model.data_pipeline.initializer)
        step_in_epoch = 0
        while True:
            try:
                start_time = time.time()
                train_result = train_model.model.train(train_sess, train_model.word_embedding)
                end_time = time.time()
                
                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)
                
                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                    train_summary_writer.add_summary(train_result.summary, global_step)
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step, "debug")
                if step_in_epoch % hyperparams.train_step_per_eval == 0 and enable_eval == True:
                    ckpt_file = infer_model.model.get_latest_ckpt("debug")
                    extrinsic_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
                        hyperparams.data_enable_preprocessing, hyperparams.train_eval_batch_size,
                        hyperparams.train_eval_metric, hyperparams.train_eval_detail_type,
                        global_step, epoch, ckpt_file, "debug")
                    decoding_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
                        hyperparams.data_enable_preprocessing, hyperparams.train_decoding_sample_size,
                        hyperparams.train_random_seed + global_step, global_step, epoch, ckpt_file, "debug")
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_summary_writer.add_summary(train_result.summary, global_step)
                train_model.model.save(train_sess, global_step, "epoch")
                if enable_eval == True:
                    ckpt_file = infer_model.model.get_latest_ckpt("epoch")
                    extrinsic_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
                        hyperparams.data_enable_preprocessing, hyperparams.train_eval_batch_size,
                        hyperparams.train_eval_metric, hyperparams.train_eval_detail_type,
                        global_step, epoch, ckpt_file, "epoch")
                    decoding_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
                        hyperparams.data_enable_preprocessing, hyperparams.train_decoding_sample_size,
                        hyperparams.train_random_seed + global_step, global_step, epoch, ckpt_file, "epoch")
                break

    train_summary_writer.close_writer()
    if enable_eval == True:
        infer_summary_writer.close_writer()
    
    logger.log_print("##### finish training #####")

def evaluate(logger,
             hyperparams,
             enable_debug=False):   
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    logger.log_print("##### create infer model #####")
    infer_model = create_infer_model(logger, hyperparams)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
    if enable_debug == True:
        infer_sess = tf_debug.LocalCLIDebugWrapperSession(infer_sess)
    
    infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
    init_model(infer_sess, infer_model)
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    
    logger.log_print("##### start evaluation #####")
    global_step = 0
    eval_mode = "debug" if enable_debug == True else "epoch"
    ckpt_file_list = infer_model.model.get_ckpt_list(eval_mode)
    for i, ckpt_file in enumerate(ckpt_file_list):
        extrinsic_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
            hyperparams.data_enable_preprocessing, hyperparams.train_eval_batch_size,
            hyperparams.train_eval_metric, hyperparams.train_eval_detail_type,
            global_step, i, ckpt_file, eval_mode)
        decoding_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
            hyperparams.data_enable_preprocessing, hyperparams.train_decoding_sample_size,
            hyperparams.train_random_seed, global_step, i, ckpt_file, eval_mode)
    
    infer_summary_writer.close_writer()
    logger.log_print("##### finish evaluation #####")

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train'):
        train(logger, hyperparams, enable_eval=False, enable_debug=False)
    elif (args.mode == 'train_eval'):
        train(logger, hyperparams, enable_eval=True, enable_debug=False)
    elif (args.mode == 'train_debug'):
        train(logger, hyperparams, enable_eval=False, enable_debug=True)
    elif (args.mode == 'eval'):
        evaluate(logger, hyperparams, enable_debug=False)
    elif (args.mode == 'eval_debug'):
        evaluate(logger, hyperparams, enable_debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
