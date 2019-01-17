import codecs
import collections
import os.path
import json

import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["DataPipeline", "create_data_pipeline",
           "create_src_data", "create_trg_data", "create_src_dataset", "create_trg_dataset",
           "generate_word_feat", "generate_subword_feat", "generate_char_feat",
           "create_embedding_file", "load_embedding_file", "convert_embedding",
           "create_vocab_file", "load_vocab_file", "process_vocab_table",
           "create_word_vocab", "create_subword_vocab", "create_char_vocab",
           "load_tsv_data", "load_json_data", "load_mrc_data",
           "prepare_data", "prepare_mrc_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "input_question_word", "input_question_subword", "input_question_char",
     "input_context_word", "input_context_subword", "input_context_char", "input_answer",
     "input_question_word_mask", "input_question_subword_mask", "input_question_char_mask",
     "input_context_word_mask", "input_context_subword_mask", "input_context_char_mask", "input_answer_mask",
     "input_question_placeholder", "input_question_word_placeholder",
     "input_question_subword_placeholder", "input_question_char_placeholder",
     "input_context_placeholder", "input_context_word_placeholder",
     "input_context_subword_placeholder", "input_context_char_placeholder",
     "input_answer_placeholder", "data_size_placeholder", "batch_size_placeholder"))):
    pass

def create_data_pipeline(input_question_word_dataset,
                         input_question_subword_dataset,
                         input_question_char_dataset,
                         input_context_word_dataset,
                         input_context_subword_dataset,
                         input_context_char_dataset,
                         input_answer_dataset,
                         input_answer_type,
                         word_vocab_index,
                         word_pad,
                         word_feat_enable,
                         subword_vocab_index,
                         subword_pad,
                         subword_feat_enable,
                         char_vocab_index,
                         char_pad,
                         char_feat_enable,
                         enable_shuffle,
                         buffer_size,
                         random_seed,
                         input_question_placeholder,
                         input_question_word_placeholder,
                         input_question_subword_placeholder,
                         input_question_char_placeholder,
                         input_context_placeholder,
                         input_context_word_placeholder,
                         input_context_subword_placeholder,
                         input_context_char_placeholder,
                         input_answer_placeholder,
                         data_size_placeholder,
                         batch_size_placeholder):
    """create data pipeline for reading comprehension model"""
    default_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    default_dataset_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
    
    if word_feat_enable == True:
        word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        word_pad_id = default_pad_id
        input_question_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        input_context_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        input_answer_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    if subword_feat_enable == True:
        subword_pad_id = tf.cast(subword_vocab_index.lookup(tf.constant(subword_pad)), dtype=tf.int32)
    else:
        subword_pad_id = default_pad_id
        input_question_subword_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        input_context_subword_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    if char_feat_enable == True:
        char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
    else:
        char_pad_id = default_pad_id
        input_question_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        input_context_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    dataset = tf.data.Dataset.zip((input_question_word_dataset, input_question_subword_dataset, input_question_char_dataset,
        input_context_word_dataset, input_context_subword_dataset, input_context_char_dataset, input_answer_dataset))
    
    if enable_shuffle == True:
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.batch(batch_size=batch_size_placeholder)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if word_feat_enable == True:
        input_question_word = batch_data[0]
        input_question_word_mask = tf.cast(tf.not_equal(batch_data[0], word_pad_id), dtype=tf.float32)
        input_context_word = batch_data[3]
        input_context_word_mask = tf.cast(tf.not_equal(batch_data[3], word_pad_id), dtype=tf.float32)
    else:
        input_question_word = None
        input_question_word_mask = None
        input_context_word = None
        input_context_word_mask = None
    
    if subword_feat_enable == True:
        input_question_subword = batch_data[1]
        input_question_subword_mask = tf.cast(tf.not_equal(batch_data[1], subword_pad_id), dtype=tf.float32)
        input_context_subword = batch_data[4]
        input_context_subword_mask = tf.cast(tf.not_equal(batch_data[4], subword_pad_id), dtype=tf.float32)
    else:
        input_question_subword = None
        input_question_subword_mask = None
        input_context_subword = None
        input_context_subword_mask = None
    
    if char_feat_enable == True:
        input_question_char = batch_data[2]
        input_question_char_mask = tf.cast(tf.not_equal(batch_data[2], char_pad_id), dtype=tf.float32)
        input_context_char = batch_data[5]
        input_context_char_mask = tf.cast(tf.not_equal(batch_data[5], char_pad_id), dtype=tf.float32)
    else:
        input_question_char = None
        input_question_char_mask = None
        input_context_char = None
        input_context_char_mask = None
    
    if input_answer_type == "span":
        index_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
        input_answer = batch_data[6]
        input_answer_mask = tf.cast(tf.greater_equal(batch_data[6], index_pad_id), dtype=tf.float32)
    elif input_answer_type == "text":
        input_answer = batch_data[6]
        input_answer_mask = tf.cast(tf.not_equal(batch_data[6], word_pad_id), dtype=tf.float32)
    else:
        input_answer = None
        input_answer_mask = None
    
    return DataPipeline(initializer=iterator.initializer,
        input_question_word=input_question_word, input_question_subword=input_question_subword,
        input_question_char=input_question_char, input_context_word=input_context_word, input_context_subword=input_context_subword,
        input_context_char=input_context_char, input_answer=input_answer, input_question_word_mask=input_question_word_mask,
        input_question_subword_mask=input_question_subword_mask, input_question_char_mask=input_question_char_mask,
        input_context_word_mask=input_context_word_mask, input_context_subword_mask=input_context_subword_mask,
        input_context_char_mask=input_context_char_mask, input_answer_mask=input_answer_mask,
        input_question_placeholder=input_question_placeholder,
        input_question_word_placeholder=input_question_word_placeholder,
        input_question_subword_placeholder=input_question_subword_placeholder,
        input_question_char_placeholder=input_question_char_placeholder,
        input_context_placeholder=input_context_placeholder,
        input_context_word_placeholder=input_context_word_placeholder,
        input_context_subword_placeholder=input_context_subword_placeholder,
        input_context_char_placeholder=input_context_char_placeholder,
        input_answer_placeholder=input_answer_placeholder,
        data_size_placeholder=data_size_placeholder, batch_size_placeholder=batch_size_placeholder)

def create_src_data(input_data,
                    word_vocab_index,
                    word_max_length,
                    word_pad,
                    word_sos,
                    word_eos,
                    word_placeholder_enable,
                    word_feat_enable,
                    subword_vocab_index,
                    subword_max_length,
                    subword_pad,
                    subword_size,
                    subword_feat_enable,
                    char_vocab_index,
                    char_max_length,
                    char_pad,
                    char_feat_enable):
    """create word/subword/char-level data for input source data"""
    word_data = None
    if word_feat_enable == True:
        word_data = [generate_word(sent, word_vocab_index, word_max_length,
            word_pad, word_sos, word_eos, word_placeholder_enable) for sent in input_data]
        word_data = np.asarray(word_data)
    
    subword_data = None
    if subword_feat_enable == True:
        subword_data = [generate_subword(sent, subword_vocab_index, word_max_length, subword_max_length,
            subword_size, word_sos, word_eos, word_placeholder_enable, subword_pad) for sent in input_data]
        subword_data = np.asarray(subword_data)

    char_data = None
    if char_feat_enable == True:
        char_data = [generate_char(sent, char_vocab_index, word_max_length, char_max_length,
            word_sos, word_eos, word_placeholder_enable, char_pad) for sent in input_data]
        char_data = np.asarray(char_data)
    
    return word_data, subword_data, char_data

def create_trg_data(input_data,
                    input_data_type,
                    word_vocab_index,
                    word_max_length,
                    word_pad,
                    word_sos,
                    word_eos,
                    word_placeholder_enable):
    """create data for input target data"""
    if input_data_type == "span":
        output_data = [span.split('|')[:2] for span in input_data]
        output_data = [np.asarray([[int(index)] for index in span]) for span in output_data]
    elif input_data_type == "text":
        output_data = [generate_word(sent, word_vocab_index, word_max_length,
            word_pad, word_sos, word_eos, word_placeholder_enable) for sent in input_data]
    
    output_data = np.asarray(output_data)
    return output_data

def generate_word(sentence,
                  word_vocab_index,
                  word_max_length,
                  word_pad,
                  word_sos,
                  word_eos,
                  word_placeholder_enable):
    """generate word feature for sentence"""
    words = sentence.strip().split(' ')
    if word_placeholder_enable == True:
        words = [word_sos] + words[:word_max_length] + [word_eos]
        word_max_length = word_max_length + 2
    else:
        words = words[:word_max_length]
    
    word_pad_index = word_vocab_index[word_pad] if word_pad in word_vocab_index else 0
    word_indices = np.full(shape=(word_max_length, 1), fill_value=word_pad_index)
    
    for i, word in enumerate(words):
        word_index = 0
        word_vocabs = [word, word.lower(), word.capitalize(), word.upper()]
        for word_vocab in word_vocabs:
            if word_vocab not in word_vocab_index:
                continue
            else:
                word_index = word_vocab_index[word_vocab]
                break
        
        word_indices[i,0] = word_index
    
    return word_indices

def generate_subword(sentence,
                     subword_vocab_index,
                     word_max_length,
                     subword_max_length,
                     subword_size,
                     word_sos,
                     word_eos,
                     word_placeholder_enable,
                     subword_pad):
    def word_to_subword(word,
                        subword_size):
        """generate subwords for word"""
        subwords = []
        chars = list(word)
        char_length = len(chars)
        for i in range(char_length-subword_size+1):
            subword =  ''.join(chars[i:i+subword_size])
            subwords.append(subword)
        
        return subwords
    
    """generate subword feature for sentence"""
    words = sentence.strip().split(' ')
    if word_placeholder_enable == True:
        words = [word_sos] + words[:word_max_length] + [word_eos]
        word_max_length = word_max_length + 2
    else:
        words = words[:word_max_length]
    
    subword_pad_index = subword_vocab_index[subword_pad] if subword_pad in subword_vocab_index else 0
    subword_indices = np.full(shape=(word_max_length, subword_max_length), fill_value=subword_pad_index)
    
    for i, word in enumerate(words):
        subwords = word_to_subword(word, subword_size)
        subwords = subwords[:subword_max_length]
        for j, subword in enumerate(subwords):
            subword_indices[i,j] = subword_vocab_index[subword] if subword in subword_vocab_index else 0
    
    return subword_indices

def generate_char(sentence,
                  char_vocab_index,
                  word_max_length,
                  char_max_length,
                  word_sos,
                  word_eos,
                  word_placeholder_enable,
                  char_pad):
    """generate char feature for sentence"""
    words = sentence.strip().split(' ')
    if word_placeholder_enable == True:
        words = [word_sos] + words[:word_max_length] + [word_eos]
        word_max_length = word_max_length + 2
    else:
        words = words[:word_max_length]
    
    char_pad_index = char_vocab_index[char_pad] if char_pad in char_vocab_index else 0
    char_indices = np.full(shape=(word_max_length, char_max_length), fill_value=char_pad_index)
    
    for i, word in enumerate(words):
        chars = list(word)
        chars = chars[:char_max_length]
        for j, char in enumerate(chars):
            char_indices[i,j] = char_vocab_index[char] if char in char_vocab_index else 0
    
    return char_indices

def create_src_dataset(input_dataset,
                       word_vocab_index,
                       word_max_length,
                       word_pad,
                       word_sos,
                       word_eos,
                       word_placeholder_enable,
                       word_feat_enable,
                       subword_vocab_index,
                       subword_max_length,
                       subword_pad,
                       subword_size,
                       subword_feat_enable,
                       char_vocab_index,
                       char_max_length,
                       char_pad,
                       char_feat_enable):
    """create word/subword/char-level dataset for input source data"""
    dataset = input_dataset
    
    word_dataset = None
    if word_feat_enable == True:
        word_dataset = dataset.map(lambda sent: generate_word_feat(sent, word_vocab_index,
            word_max_length, word_pad, word_sos, word_eos, word_placeholder_enable))

    subword_dataset = None
    if subword_feat_enable == True:
        subword_dataset = dataset.map(lambda sent: generate_subword_feat(sent, subword_vocab_index,
            word_max_length, subword_max_length, subword_size, word_sos, word_eos, word_placeholder_enable, subword_pad))

    char_dataset = None
    if char_feat_enable == True:
        char_dataset = dataset.map(lambda sent: generate_char_feat(sent, char_vocab_index,
            word_max_length, char_max_length, word_sos, word_eos, word_placeholder_enable, char_pad))
    
    return word_dataset, subword_dataset, char_dataset

def create_trg_dataset(input_dataset,
                       input_data_type,
                       word_vocab_index,
                       word_max_length,
                       word_pad,
                       word_sos,
                       word_eos,
                       word_placeholder_enable):
    """create dataset for input target data"""
    dataset = input_dataset
    
    if input_data_type == "span":
        dataset = dataset.map(lambda span: tf.string_split([span], delimiter='|').values)
        dataset = dataset.map(lambda span: tf.string_to_number(span, out_type=tf.int32))
        dataset = dataset.map(lambda span: tf.expand_dims(span, axis=-1))
    elif input_data_type == "text":
        dataset = dataset.map(lambda sent: generate_word_feat(sent, word_vocab_index,
            word_max_length, word_pad, word_sos, word_eos, word_placeholder_enable))
    
    return dataset

def generate_word_feat(sentence,
                       word_vocab_index,
                       word_max_length,
                       word_pad,
                       word_sos,
                       word_eos,
                       word_placeholder_enable):
    """generate word feature for sentence"""
    words = tf.string_split([sentence], delimiter=' ').values
    if word_placeholder_enable == True:
        words = tf.concat([[word_sos], words[:word_max_length], [word_eos],
            tf.constant(word_pad, shape=[word_max_length])], axis=0)
        word_max_length = word_max_length + 2
    else:
        words = tf.concat([words[:word_max_length],
            tf.constant(word_pad, shape=[word_max_length])], axis=0)
    words = tf.reshape(words[:word_max_length], shape=[word_max_length])
    words = tf.cast(word_vocab_index.lookup(words), dtype=tf.int32)
    words = tf.expand_dims(words, axis=-1)
    
    return words

def generate_subword_feat(sentence,
                          subword_vocab_index,
                          word_max_length,
                          subword_max_length,
                          subword_size,
                          word_sos,
                          word_eos,
                          word_placeholder_enable,
                          subword_pad):
    def word_to_subword(word):
        """generate subwords for word"""
        word_len = tf.size(tf.string_split([word], delimiter=''))
        subwords = tf.substr([word], 0, subword_size)
        for i in range(1, subword_max_length):
            subwords = tf.cond(i+subword_size-1 < word_len,
                lambda: tf.concat([subwords, tf.substr([word], i, subword_size)], 0),
                lambda: subwords)
        
        subwords = tf.concat([subwords[:subword_max_length],
            tf.constant(subword_pad, shape=[subword_max_length])], axis=0)
        subwords = tf.reshape(subwords[:subword_max_length], shape=[subword_max_length])
        
        return subwords
    
    """generate subword feature for sentence"""
    words = tf.string_split([sentence], delimiter=' ').values
    if word_placeholder_enable == True:
        words = tf.concat([[word_sos], words[:word_max_length], [word_eos],
            tf.constant(subword_pad, shape=[word_max_length])], axis=0)
        word_max_length = word_max_length + 2
    else:
        words = tf.concat([words[:word_max_length],
            tf.constant(subword_pad, shape=[word_max_length])], axis=0)
    
    words = tf.reshape(words[:word_max_length], shape=[word_max_length])
    word_subwords = tf.map_fn(word_to_subword, words)
    word_subwords = tf.cast(subword_vocab_index.lookup(word_subwords), dtype=tf.int32)
    
    return word_subwords

def generate_char_feat(sentence,
                       char_vocab_index,
                       word_max_length,
                       char_max_length,
                       word_sos,
                       word_eos,
                       word_placeholder_enable,
                       char_pad):
    def word_to_char(word):
        """generate chars for word"""
        chars = tf.string_split([word], delimiter='').values
        chars = tf.concat([chars[:char_max_length],
            tf.constant(char_pad, shape=[char_max_length])], axis=0)
        chars = tf.reshape(chars[:char_max_length], shape=[char_max_length])
        
        return chars
    
    """generate char feature for sentence"""
    words = tf.string_split([sentence], delimiter=' ').values
    if word_placeholder_enable == True:
        words = tf.concat([[word_sos], words[:word_max_length], [word_eos],
            tf.constant(char_pad, shape=[word_max_length])], axis=0)
        word_max_length = word_max_length + 2
    else:
        words = tf.concat([words[:word_max_length],
            tf.constant(char_pad, shape=[word_max_length])], axis=0)
    
    words = tf.reshape(words[:word_max_length], shape=[word_max_length])
    word_chars = tf.map_fn(word_to_char, words)
    word_chars = tf.cast(char_vocab_index.lookup(word_chars), dtype=tf.int32)
    
    return word_chars

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not os.path.exists(embedding_dir):
        os.path.mkdir(embedding_dir)
    
    if not os.path.exists(embedding_file):
        with open(embedding_file, 'wb') as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str).encode('utf-8'))

def load_embedding_file(embedding_file,
                        embedding_size,
                        unk,
                        pad,
                        sos,
                        eos):
    """load pre-train embeddings from embedding file"""
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as file:
            embedding = {}
            for line in file:
                items = line.decode('utf-8').strip().split(' ')
                if len(items) != embedding_size + 1:
                    continue
                word = items[0]
                vector = [float(x) for x in items[1:]]
                if word not in embedding:
                    embedding[word] = vector
            
            if unk not in embedding:
                embedding[unk] = np.random.rand(embedding_size)
            if pad not in embedding:
                embedding[pad] = np.random.rand(embedding_size)
            if sos and sos not in embedding:
                embedding[sos] = np.random.rand(embedding_size)
            if eos and eos not in embedding:
                embedding[eos] = np.random.rand(embedding_size)
            
            return embedding
    else:
        raise FileNotFoundError("embedding file not found")

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

def create_vocab_file(vocab_file,
                      vocab_table):
    """create vocab file based on vocab table"""
    vocab_dir = os.path.dirname(vocab_file)
    if not os.path.exists(vocab_dir):
        os.path.mkdir(vocab_dir)
    
    if not os.path.exists(vocab_file):
        with open(vocab_file, 'wb') as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab).encode('utf-8'))

def load_vocab_file(vocab_file):
    """load vocab data from vocab file"""
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as file:
            vocab = {}
            for line in file:
                items = line.decode('utf-8').strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = MAX_INT
            
            return vocab
    else:
        raise FileNotFoundError("vocab file not found")

def process_vocab_table(vocab,
                        vocab_size,
                        vocab_threshold,
                        vocab_lookup,
                        unk,
                        pad,
                        sos,
                        eos):
    """process vocab table"""
    default_vocab = [unk, pad]
    if sos:
        default_vocab.append(sos)
    if eos:
        default_vocab.append(eos)
    
    if unk in vocab:
        del vocab[unk]
    if pad in vocab:
        del vocab[pad]
    if sos and sos in vocab:
        del vocab[sos]
    if eos and eos in vocab:
        del vocab[eos]
    
    vocab = { k: vocab[k] for k in vocab.keys() if vocab[k] >= vocab_threshold }
    if vocab_lookup is not None:
        vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
    
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    sorted_vocab = default_vocab + sorted_vocab
    
    vocab_table = sorted_vocab[:vocab_size]
    vocab_size = len(vocab_table)
    
    vocab_index = {vocab: int(idx) for idx, vocab in enumerate(vocab_table)}
    vocab_inverted_index = {vocab_index[vocab]: vocab for vocab in vocab_index.keys()}
    
    return vocab_table, vocab_size, vocab_index, vocab_inverted_index

def create_word_vocab(input_data):
    """create word vocab from input data"""
    word_vocab_lookup = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            word_vocabs = [word, word.lower(), word.capitalize(), word.upper()]
            for word_vocab in word_vocabs:
                if word_vocab not in word_vocab_lookup:
                    word_vocab_lookup[word_vocab] = 1
                else:
                    word_vocab_lookup[word_vocab] += 1
    
    return word_vocab_lookup

def create_subword_vocab(input_data,
                         subword_size):
    """create subword vocab from input data"""
    def generate_subword(word,
                         subword_size):
        """generate subword for word"""
        subwords = []
        chars = list(word)
        char_length = len(chars)
        for i in range(char_length-subword_size+1):
            subword =  ''.join(chars[i:i+subword_size])
            subwords.append(subword)
        
        return subwords
    
    subword_vocab_lookup = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            word_vocabs = [word, word.lower(), word.capitalize(), word.upper()]
            for word_vocab in word_vocabs:
                subword_vocabs = generate_subword(word_vocab, subword_size)
                for subword_vocab in subword_vocabs:
                    if subword_vocab not in subword_vocab_lookup:
                        subword_vocab_lookup[subword_vocab] = 1
                    else:
                        subword_vocab_lookup[subword_vocab] += 1
    
    return subword_vocab_lookup

def create_char_vocab(input_data):
    """create char vocab from input data"""
    char_vocab_lookup = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            word_vocabs = [word, word.lower(), word.capitalize(), word.upper()]
            for word_vocab in word_vocabs:
                char_vocabs = list(word_vocab)
                for char_vocab in char_vocabs:
                    if char_vocab not in char_vocab_lookup:
                        char_vocab_lookup[char_vocab] = 1
                    else:
                        char_vocab_lookup[char_vocab] += 1
    
    return char_vocab_lookup

def load_tsv_data(input_file):
    """load data from tsv file"""
    if os.path.exists(input_file):
        with open(input_file, 'rb') as file:
            input_data = []
            for line in file:
                input_data.append(line.decode('utf-8').strip())
            
            return input_data
    else:
        raise FileNotFoundError("input file not found")

def load_json_data(input_file):
    """load data from json file"""
    if os.path.exists(input_file):
        with open(input_file, 'r') as file:
            input_data = json.load(file)
            
            return input_data
    else:
        raise FileNotFoundError("input file not found")
    
def load_mrc_data(input_file,
                  file_type,
                  answer_type,
                  expand_multiple_answer):
    """load mrc data from input file"""
    question_data = []
    context_data = []
    answer_text_data = []
    answer_span_data = []
    if file_type == "tsv":
        input_data = load_tsv_data(input_file)
        for line in input_data:
            item = line.split("\t")
            question_data.append(item[1])
            context_data.append(item[2])
            answer_text_data.append(item[3])
            answer_span_data.append(item[4])
    elif file_type == "json":
        input_data = load_json_data(input_file)
        for item in input_data:
            if expand_multiple_answer == True:
                for answer in item["answers"]:
                    question_data.append(item["question"])
                    context_data.append(item["context"])
                    answer_text_data.append(answer["text"])
                    answer_span_data.append("{0}|{1}".format(answer["start"], answer["end"]))
            else:
                question_data.append(item["question"])
                context_data.append(item["context"])
                answer_text_data.append(item["answers"][0]["text"])
                answer_span_data.append("{0}|{1}".format(item["answers"][0]["start"], item["answers"][0]["end"]))
    else:
        raise ValueError("can not load data from unsupported file type {0}".format(file_type))
    
    if answer_type == "text":
        answer_data = answer_text_data
    elif answer_type == "span":
        answer_data = answer_span_data
    else:
        raise ValueError("can not genertate data from unsupported answer type {0}".format(answer_type))
    
    return input_data, question_data, context_data, answer_data

def validate_data(logger,
                  input_mrc_data,
                  input_question_data,
                  input_context_data,
                  input_answer_data,
                  max_question_length,
                  max_context_length,
                  max_answer_length,
                  answer_type):
    output_mrc_data = []
    output_question_data = []
    output_context_data = []
    output_answer_data = []
    
    input_data = zip(input_mrc_data, input_question_data, input_context_data, input_answer_data)
    for input_mrc, input_question, input_context, input_answer in input_data:
        question_length = len(input_question.split(' '))
        context_length = len(input_context.split(' '))
        
        if answer_type == "text":
            answer_length = len(input_answer.split(' '))
            max_answer_index = -1
        elif answer_type == "span":
            answer_span = input_answer.split('|')
            answer_start = int(answer_span[0].strip())
            answer_end = int(answer_span[1].strip())
            answer_length = answer_end - answer_start + 1
            max_answer_index = max(answer_start, answer_end)
        
        if (question_length > max_question_length or 
            context_length > max_context_length or 
            answer_length > max_answer_length or
            max_answer_index >= max_context_length):
            continue
        
        output_mrc_data.append(input_mrc)
        output_question_data.append(input_question)
        output_context_data.append(input_context)
        output_answer_data.append(input_answer)
    
    return output_mrc_data, output_question_data, output_context_data, output_answer_data

def prepare_data(logger,
                 input_data,
                 word_vocab_file,
                 word_vocab_size,
                 word_vocab_threshold,
                 word_embed_dim,
                 word_embed_file,
                 full_word_embed_file,
                 word_unk,
                 word_pad,
                 word_sos,
                 word_eos,
                 word_feat_enable,
                 pretrain_word_embed,
                 subword_vocab_file,
                 subword_vocab_size,
                 subword_vocab_threshold,
                 subword_unk,
                 subword_pad,
                 subword_size,
                 subword_feat_enable,
                 char_vocab_file,
                 char_vocab_size,
                 char_vocab_threshold,
                 char_unk,
                 char_pad,
                 char_feat_enable):
    """prepare data"""    
    word_embed_data = None
    if pretrain_word_embed == True:
        if os.path.exists(word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_file))
            word_embed_data = load_embedding_file(word_embed_file,
                word_embed_dim, word_unk, word_pad, word_sos, word_eos)
        elif os.path.exists(full_word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(full_word_embed_file))
            word_embed_data = load_embedding_file(full_word_embed_file,
                word_embed_dim, word_unk, word_pad, word_sos, word_eos)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_vocab_file, full_word_embed_file))
        
        word_embed_size = len(word_embed_data) if word_embed_data is not None else 0
        logger.log_print("# word embedding table has {0} words".format(word_embed_size))
    
    word_vocab = None
    word_vocab_index = None
    word_vocab_inverted_index = None
    if os.path.exists(word_vocab_file):
        logger.log_print("# loading word vocab table from {0}".format(word_vocab_file))
        word_vocab = load_vocab_file(word_vocab_file)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad, word_sos, word_eos)
    elif input_data is not None:
        logger.log_print("# creating word vocab table from input data")
        word_vocab = create_word_vocab(input_data)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad, word_sos, word_eos)
        logger.log_print("# creating word vocab file {0}".format(word_vocab_file))
        create_vocab_file(word_vocab_file, word_vocab_table)
    else:
        raise ValueError("{0} or input data must be provided".format(word_vocab_file))
    
    logger.log_print("# word vocab table has {0} words".format(word_vocab_size))
    
    subword_vocab = None
    subword_vocab_index = None
    subword_vocab_inverted_index = None
    if subword_feat_enable is True:
        if os.path.exists(subword_vocab_file):
            logger.log_print("# loading subword vocab table from {0}".format(subword_vocab_file))
            subword_vocab = load_vocab_file(subword_vocab_file)
            (_, subword_vocab_size, subword_vocab_index,
                subword_vocab_inverted_index) = process_vocab_table(subword_vocab, subword_vocab_size,
                subword_vocab_threshold, None, subword_unk, subword_pad, None, None)
        elif input_data is not None:
            logger.log_print("# creating subword vocab table from input data")
            subword_vocab = create_subword_vocab(input_data, subword_size)
            (subword_vocab_table, subword_vocab_size, subword_vocab_index,
                subword_vocab_inverted_index) = process_vocab_table(subword_vocab, subword_vocab_size,
                subword_vocab_threshold, None, subword_unk, subword_pad, None, None)
            logger.log_print("# creating subword vocab file {0}".format(subword_vocab_file))
            create_vocab_file(subword_vocab_file, subword_vocab_table)
        else:
            raise ValueError("{0} or input data must be provided".format(subword_vocab_file))

        logger.log_print("# subword vocab table has {0} subwords".format(subword_vocab_size))
    
    char_vocab = None
    char_vocab_index = None
    char_vocab_inverted_index = None
    if char_feat_enable is True:
        if os.path.exists(char_vocab_file):
            logger.log_print("# loading char vocab table from {0}".format(char_vocab_file))
            char_vocab = load_vocab_file(char_vocab_file)
            (_, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad, None, None)
        elif input_data is not None:
            logger.log_print("# creating char vocab table from input data")
            char_vocab = create_char_vocab(input_data)
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad, None, None)
            logger.log_print("# creating char vocab file {0}".format(char_vocab_file))
            create_vocab_file(char_vocab_file, char_vocab_table)
        else:
            raise ValueError("{0} or input data must be provided".format(char_vocab_file))

        logger.log_print("# char vocab table has {0} chars".format(char_vocab_size))
    
    if word_embed_data is not None and word_vocab_table is not None:
        word_embed_data = { k: word_embed_data[k] for k in word_vocab_table if k in word_embed_data }
        logger.log_print("# word embedding table has {0} words after filtering".format(len(word_embed_data)))
        if not os.path.exists(word_embed_file):
            logger.log_print("# creating word embedding file {0}".format(word_embed_file))
            create_embedding_file(word_embed_file, word_embed_data)
        
        word_embed_data = convert_embedding(word_embed_data)
    
    return (word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)

def prepare_mrc_data(logger,
                     input_mrc_file,
                     input_file_type,
                     input_answer_type,
                     input_expand_multiple_answer,
                     max_question_length,
                     max_context_length,
                     max_answer_length,
                     enable_validation,
                     word_vocab_file,
                     word_vocab_size,
                     word_vocab_threshold,
                     word_embed_dim,
                     word_embed_file,
                     full_word_embed_file,
                     word_unk,
                     word_pad,
                     word_sos,
                     word_eos,
                     word_feat_enable,
                     pretrain_word_embed,
                     subword_vocab_file,
                     subword_vocab_size,
                     subword_vocab_threshold,
                     subword_unk,
                     subword_pad,
                     subword_size,
                     subword_feat_enable,
                     char_vocab_file,
                     char_vocab_size,
                     char_vocab_threshold,
                     char_unk,
                     char_pad,
                     char_feat_enable):
    """prepare mrc data"""
    logger.log_print("# loading input mrc data from {0}".format(input_mrc_file))
    (input_mrc_data, input_question_data, input_context_data,
        input_answer_data) = load_mrc_data(input_mrc_file,
            input_file_type, input_answer_type, input_expand_multiple_answer)
    
    input_mrc_size = len(input_mrc_data)
    input_question_size = len(input_question_data)
    input_context_size = len(input_context_data)
    input_answer_size = len(input_answer_data)
    logger.log_print("# input mrc data has {0} lines".format(input_mrc_size))
    
    if input_answer_size != input_question_size or input_answer_size != input_context_size:
        raise ValueError("question, context & answer input data must have the same size")
    
    if enable_validation == True:
        (input_mrc_data, input_question_data, input_context_data,
            input_answer_data) = validate_data(logger, input_mrc_data, input_question_data, input_context_data, input_answer_data,
            max_question_length, max_context_length, max_answer_length, input_answer_type)
        input_mrc_size = len(input_mrc_data)
        logger.log_print("# input mrc data has {0} lines after validation".format(input_mrc_size))
    
    input_text_data = set()
    input_text_data.update(input_question_data)
    input_text_data.update(input_context_data)
    if input_answer_type == "text":
        input_text_data.update(input_answer_data)
    
    input_text_data = list(input_text_data)
    (word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_data(logger, input_text_data,
            word_vocab_file, word_vocab_size, word_vocab_threshold, word_embed_dim, word_embed_file,
            full_word_embed_file, word_unk, word_pad, word_sos, word_eos, word_feat_enable, pretrain_word_embed,
            subword_vocab_file, subword_vocab_size, subword_vocab_threshold, subword_unk, subword_pad, subword_size,
            subword_feat_enable, char_vocab_file, char_vocab_size, char_vocab_threshold, char_unk, char_pad, char_feat_enable)
    
    return (input_mrc_data, input_question_data, input_context_data, input_answer_data,
        word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)
