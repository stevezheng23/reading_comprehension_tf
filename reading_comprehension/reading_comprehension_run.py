import argparse
import json

import numpy as np
import tensorflow as tf

from util.debug_logger import *
from util.default_util import *
from util.param_util import *
from util.data_util import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def train(logger,
          hyperparams):
    pass

def evaluate(logger,
             hyperparams):
    pass

def test(logger):   
    graph = tf.Graph()
    with graph.as_default():       
        (input_data, word_embed_data,
             word_vocab_size, word_vocab_index, word_vocab_inverted_index,
             subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
             char_vocab_size, char_vocab_index,
             char_vocab_inverted_index) = prepare_data(logger,
                                                       input_file="data/CWMT_NEU/NEU_en.token.test.txt",
                                                       word_vocab_file="data/CWMT_NEU/NEU_en.vocab_word.txt",
                                                       word_vocab_size=30000,
                                                       word_embed_dim=300,
                                                       word_embed_file="data/CWMT_NEU/cwmt_neu2017.wiki.en.vec",
                                                       full_word_embed_file="",
                                                       word_unk="<unk>",
                                                       word_pad="<pad>",
                                                       word_sos="<sos>",
                                                       word_eos="<eos>",
                                                       word_feat_enable=True,
                                                       pretrain_word_embed=True,
                                                       subword_vocab_file="data/CWMT_NEU/NEU_en.vocab_subword.txt",
                                                       subword_vocab_size=10000,
                                                       subword_unk="<u>",
                                                       subword_pad="<p>",
                                                       subword_size=3,
                                                       subword_feat_enable=True,
                                                       char_vocab_file="data/CWMT_NEU/NEU_en.vocab_char.txt",
                                                       char_vocab_size=1000,
                                                       char_unk="<u>",
                                                       char_pad="<p>",
                                                       char_feat_enable=True)
        
        (input_src_word_dataset, input_src_subword_dataset,
            input_src_char_dataset) = create_input_dataset(input_file="data/CWMT_NEU/NEU_en.token.test.txt",
                                                           word_vocab_index=word_vocab_index,
                                                           word_max_length=20,
                                                           word_pad="<pad>",
                                                           word_sos="<sos>",
                                                           word_eos="<eos>",
                                                           word_feat_enable=True,
                                                           subword_vocab_index=subword_vocab_index,
                                                           subword_max_length=10,
                                                           subword_size=3,
                                                           subword_pad="<p>",
                                                           subword_feat_enable=True,
                                                           char_vocab_index=char_vocab_index,
                                                           char_max_length=10,
                                                           char_pad="<p>",
                                                           char_feat_enable=True)
        
        (input_trg_word_dataset, input_trg_subword_dataset,
            input_trg_char_dataset) = create_input_dataset(input_file="data/CWMT_NEU/NEU_en.token.test.txt",
                                                           word_vocab_index=word_vocab_index,
                                                           word_max_length=20,
                                                           word_pad="<pad>",
                                                           word_sos="<sos>",
                                                           word_eos="<eos>",
                                                           word_feat_enable=True,
                                                           subword_vocab_index=subword_vocab_index,
                                                           subword_max_length=10,
                                                           subword_size=3,
                                                           subword_pad="<p>",
                                                           subword_feat_enable=True,
                                                           char_vocab_index=char_vocab_index,
                                                           char_max_length=10,
                                                           char_pad="<p>",
                                                           char_feat_enable=True)
        
        output_label_word_dataset = create_output_dataset(output_file="data/CWMT_NEU/NEU_en.token.test.txt",
                                                          word_vocab_index=word_vocab_index,
                                                          word_max_length=30,
                                                          word_sos="<sos>",
                                                          word_eos="<eos>",
                                                          word_feat_enable=True)
        
        data_pipeline = create_data_pipeline(input_src_word_dataset=input_src_word_dataset,
                                             input_src_subword_dataset=input_src_subword_dataset,
                                             input_src_char_dataset=input_src_char_dataset,
                                             input_trg_word_dataset=input_trg_word_dataset,
                                             input_trg_subword_dataset=input_trg_subword_dataset,
                                             input_trg_char_dataset=input_trg_char_dataset,
                                             output_label_word_dataset=output_label_word_dataset,
                                             word_vocab_index=word_vocab_index,
                                             word_pad="<pad>",
                                             word_feat_enable=True,
                                             subword_vocab_index=subword_vocab_index,
                                             subword_pad="<p>",
                                             subword_feat_enable=True,
                                             char_vocab_index=char_vocab_index,
                                             char_pad="<p>",
                                             char_feat_enable=True,
                                             dataset_size=len(input_data),
                                             batch_size=2,
                                             random_seed=100,
                                             enable_shuffle=False)
        
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(data_pipeline.initializer)
            
            (input_source_word, input_source_subword,
                input_source_char) = sess.run([data_pipeline.input_source_word,
                    data_pipeline.input_source_subword, data_pipeline.input_source_char])
            print(input_source_word)
            print(input_source_subword)
            print(input_source_char)

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
        test(logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
