import numpy as np
import tensorflow as tf

from external.bleu import *
from external.rouge import *
from external.squad import *

__all__ = ["evaluate_from_data", "evaluate_from_file"]

def _bleu(pred_data, ref_data):
    """BLEU score for translation task"""
    max_order = 4
    smooth = False
    score, _, _, _, _, _ = compute_bleu(ref_data, pred_data, max_order, smooth)
    bleu_score = 100 * score
    return bleu_score

def _rouge(pred_data, ref_data):
    """ROUGE score for summarization task"""
    score_map = rouge(pred_data, ref_data)
    rouge_score = 100 * score_map["rouge_l/f_score"]
    return rouge_score

def _squad_em(pred_data, ref_data):
    """EM score for reading comprehension task"""
    em_score = eval_exact_match_score(pred_data, ref_data)
    return em_score

def _squad_f1(pred_data, ref_data):
    """F1 score for reading comprehension task"""
    f1_score = eval_f1_score(pred_data, ref_data)
    return f1_score

def evaluate_from_data(pred_data, ref_data, metric):
    """compute evaluation score based on selected metric"""
    pred_and_ref = [(pred, ref_list) for pred, ref_list in zip(pred_data, ref_data) if pred and ref_list]
    pred_data = [pred for (pred, _) in pred_and_ref]
    ref_data = [ref_list for (_, ref_list) in pred_and_ref]
    
    if len(pred_data) == 0 or len(ref_data) == 0:
        return 0.0
    
    if metric == "bleu":
        eval_score = _bleu(pred_data, ref_data)
    elif metric == "rouge":
        eval_score = _rouge(pred_data, ref_data)
    elif metric == "exact":
        eval_score = _squad_em(pred_data, ref_data)
    elif metric == "f1":
        eval_score = _squad_f1(pred_data, ref_data)
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
