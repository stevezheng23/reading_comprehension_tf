import numpy as np
import tensorflow as tf

from external.bleu import *
from external.rouge import *

__all__ = ["evaluate_from_data", "evaluate_from_file"]

def _bleu(pred_data, ref_data):
    """BLEU score for translation task"""
    max_order = 4
    smooth = False
    ref_list_data = [[ref] for ref in ref_data]
    score, _, _, _, _, _ = compute_bleu(ref_list_data, pred_data, max_order, smooth)
    bleu_score = 100 * score
    return bleu_score

def _rouge(pred_data, ref_data):
    """ROUGE score for summarization task"""
    score_map = rouge(pred_data, ref_data)
    rouge_score = 100 * score_map["rouge_l/f_score"]
    return rouge_score

def _accuracy(pred_data, ref_data):
    """compute sentence-level accuracy"""
    pred_size = len(pred_data)
    ref_size = len(ref_data)
    if pred_size <= 0 or ref_size <= 0:
        raise ValueError("size of predict or reference data is less than or equal to 0")
    if pred_size != ref_size:
        raise ValueError("size of predict and reference data don't match")
    
    match_count = 0
    total_count = 0
    for i in range(pred_size):
        predict = pred_data[i]
        reference = ref_data[i]
        if predict == reference:
            match_count += 1
        total_count += 1
    
    accuracy = 100.0 * match_count / total_count
    return accuracy

def _word_accuracy(pred_data, ref_data):
    """compute word-level accuracy"""
    pred_size = len(pred_data)
    ref_size = len(ref_data)
    if pred_size <= 0 or ref_size <= 0:
        raise ValueError("size of predict or reference data is less than or equal to 0")
    if pred_size != ref_size:
        raise ValueError("size of predict and reference data don't match")
    
    total_count = 0
    for i in range(pred_size):
        pred_word = pred_data[i].strip().slipt(" ")
        ref_word = ref_data[i].strip().slipt(" ")
        pred_len = len(pred_word)
        ref_len = len(ref_word)
        match_count = 0
        for j in range(min(pred_len, ref_len)):
            predict_word = pred_word[j]
            reference_word = ref_word[j]
            if predict_word == reference_word:
                match_count += 1
        total_accuracy += 100.0 * match_count / max(pred_len, ref_len)
        total_count += 1
    
    word_accuracy = total_accuracy / total_count
    return word_accuracy

def evaluate_from_data(pred_data, ref_data, metric):
    """compute evaluation score based on selected metric"""
    pred_and_ref = [(pred, ref) for pred, ref in zip(pred_data, ref_data) if pred and ref]
    pred_data = [pred for (pred, ref) in pred_and_ref]
    ref_data = [ref.encode("utf-8") for (pred, ref) in pred_and_ref]
    
    if len(pred_data) == 0 or len(ref_data) == 0:
        return 0.0
    
    if metric == "bleu":
        eval_score = _bleu(pred_data, ref_data)
    elif metric == "rouge":
        eval_score = _rouge(pred_data, ref_data)
    elif metric == "accuracy":
        eval_score = _accuracy(pred_data, ref_data)
    elif metric == "word_accuracy":
        eval_score = _word_accuracy(pred_data, ref_data)
    else:
        raise ValueError("unsupported metric {0}".format(metric))
    
    return eval_score

def evaluate_from_file(pred_file, ref_file, metric):
    predict = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as file_p:
        for line in file_p:
            predict.append(line.strip())
    reference = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as file_r:
        for line in file_r:
            reference.append(line.strip())
    
    eval_score = evaluate(predict, reference, metric)
    return eval_score
