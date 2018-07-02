import argparse
import codecs
import json
import math
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["create_default_hyperparams", "load_hyperparams",
           "generate_search_lookup", "search_hyperparams", "create_hyperparams_file"]

def create_default_hyperparams(config_type):
    """create default hyperparameters"""
    if config_type == "bidaf":
        hyperparams = tf.contrib.training.HParams(
            data_train_mrc_file="",
            data_train_mrc_file_type="",
            data_eval_mrc_file="",
            data_eval_mrc_file_type="",
            data_embedding_file="",
            data_full_embedding_file="",
            data_max_question_length=40,
            data_max_context_length=500,
            data_max_answer_length=30,
            data_max_subword_length=16,
            data_max_char_length=16,
            data_word_vocab_file="",
            data_word_vocab_size=50000,
            data_word_vocab_threshold=10,
            data_word_unk="<unk>",
            data_word_pad="<pad>",
            data_word_sos="<sos>",
            data_word_eos="<eos>",
            data_word_placeholder_enable=False,
            data_subword_vocab_file="",
            data_subword_vocab_size=20000,
            data_subword_vocab_threshold=20,
            data_subword_unk="***",
            data_subword_pad="###",
            data_subword_size=3,
            data_char_vocab_file="",
            data_char_vocab_size=1000,
            data_char_vocab_threshold=50,
            data_char_unk="*",
            data_char_pad="#",
            data_answer_type="",
            data_expand_multiple_answer=False,
            data_log_output_dir="",
            data_result_output_dir="",
            train_random_seed=100,
            train_enable_shuffle=True,
            train_batch_size=60,
            train_eval_batch_size=100,
            train_eval_metric=["exact", "f1"],
            train_eval_detail_type="full",
            train_decoding_sample_size=3,
            train_num_epoch=3,
            train_ckpt_output_dir="",
            train_summary_output_dir="",
            train_step_per_stat=10,
            train_step_per_ckpt=1000,
            train_step_per_eval=1000,
            train_clip_norm=5.0,
            train_ema_enable=True,
            train_ema_decay_rate=0.999,
            train_optimizer_type="adam",
            train_optimizer_learning_rate=0.001,
            train_optimizer_decay_enable=False,
            train_optimizer_decay_mode="exponential_decay",
            train_optimizer_decay_rate=0.95,
            train_optimizer_decay_step=1000,
            train_optimizer_decay_start_step=10000,
            train_optimizer_momentum_beta=0.9,
            train_optimizer_rmsprop_beta=0.999,
            train_optimizer_rmsprop_epsilon=1e-8,
            train_optimizer_adadelta_rho=0.95,
            train_optimizer_adadelta_epsilon=1e-8,
            train_optimizer_adagrad_init_accumulator=0.1,
            train_optimizer_adam_beta_1=0.9,
            train_optimizer_adam_beta_2=0.999,
            train_optimizer_adam_epsilon=1e-08,
            model_type="bidaf",
            model_scope="mrc",
            model_representation_word_embed_dim=100,
            model_representation_word_embed_pretrained=True,
            model_representation_word_feat_trainable=False,
            model_representation_word_feat_enable=True,
            model_representation_subword_embed_dim=8,
            model_representation_subword_unit_dim=100,
            model_representation_subword_window_size=[5],
            model_representation_subword_hidden_activation="relu",
            model_representation_subword_dropout=0.2,
            model_representation_subword_pooling_type="max",
            model_representation_subword_feat_trainable=True,
            model_representation_subword_feat_enable=True,
            model_representation_char_embed_dim=8,
            model_representation_char_unit_dim=100,
            model_representation_char_window_size=[5],
            model_representation_char_hidden_activation="relu",
            model_representation_char_dropout=0.2,
            model_representation_char_pooling_type="max",
            model_representation_char_feat_trainable=True,
            model_representation_char_feat_enable=True,
            model_representation_fusion_type="highway",
            model_representation_fusion_num_layer=2,
            model_representation_fusion_unit_dim=100,
            model_representation_fusion_hidden_activation="relu",
            model_representation_fusion_dropout=0.2,
            model_representation_fusion_trainable=True,
            model_understanding_question_num_layer=1,
            model_understanding_question_unit_dim=100,
            model_understanding_question_cell_type="lstm",
            model_understanding_question_hidden_activation="tanh",
            model_understanding_question_dropout=0.2,
            model_understanding_question_forget_bias=False,
            model_understanding_question_residual_connect=False,
            model_understanding_question_trainable=True,
            model_understanding_context_num_layer=1,
            model_understanding_context_unit_dim=100,
            model_understanding_context_cell_type="lstm",
            model_understanding_context_hidden_activation="tanh",
            model_understanding_context_dropout=0.2,
            model_understanding_context_forget_bias=False,
            model_understanding_context_residual_connect=False,
            model_understanding_context_trainable=True,
            model_understanding_enable_sharing=True,
            model_interaction_context2quesiton_attention_dim=200,
            model_interaction_context2quesiton_score_type="linear_plus",
            model_interaction_context2quesiton_trainable=True,
            model_interaction_context2quesiton_enable=True,
            model_interaction_quesiton2context_attention_dim=200,
            model_interaction_quesiton2context_score_type="linear_plus",
            model_interaction_quesiton2context_trainable=True,
            model_interaction_quesiton2context_enable=True,
            model_interaction_fusion_type="concate",
            model_interaction_fusion_num_layer=1,
            model_interaction_fusion_unit_dim=800,
            model_interaction_fusion_hidden_activation="relu",
            model_interaction_fusion_dropout=0.2,
            model_interaction_fusion_trainable=True,
            model_interaction_fusion_combo_enable=True,
            model_interaction_enable_sharing=True,
            model_modeling_answer_num_layer=1,
            model_modeling_answer_unit_dim=100,
            model_modeling_answer_cell_type="lstm",
            model_modeling_answer_hidden_activation="tanh",
            model_modeling_answer_dropout=0.2,
            model_modeling_answer_forget_bias=False,
            model_modeling_answer_residual_connect=False,
            model_modeling_answer_attention_dim=200,
            model_modeling_answer_score_type="linear_plus",
            model_modeling_answer_attention_enable=False,
            model_modeling_answer_trainable=True,
            model_modeling_fusion_type="concate",
            model_modeling_fusion_num_layer=1,
            model_modeling_fusion_unit_dim=200,
            model_modeling_fusion_hidden_activation="relu",
            model_modeling_fusion_dropout=0.2,
            model_modeling_fusion_trainable=True,
            model_output_answer_start_num_layer=1,
            model_output_answer_start_unit_dim=100,
            model_output_answer_start_cell_type="lstm",
            model_output_answer_start_hidden_activation="tanh",
            model_output_answer_start_dropout=0.2,
            model_output_answer_start_forget_bias=False,
            model_output_answer_start_residual_connect=False,
            model_output_answer_start_trainable=True,
            model_output_answer_end_num_layer=1,
            model_output_answer_end_unit_dim=100,
            model_output_answer_end_cell_type="lstm",
            model_output_answer_end_hidden_activation="tanh",
            model_output_answer_end_dropout=0.2,
            model_output_answer_end_forget_bias=False,
            model_output_answer_end_residual_connect=False,
            model_output_answer_end_trainable=True,
            device_num_gpus=1,
            device_default_gpu_id=0,
            device_log_device_placement=False,
            device_allow_soft_placement=False,
            device_allow_growth=False,
            device_per_process_gpu_memory_fraction=0.8
        )
    elif config_type == "qanet":
        hyperparams = tf.contrib.training.HParams(
            data_train_mrc_file="",
            data_train_mrc_file_type="",
            data_eval_mrc_file="",
            data_eval_mrc_file_type="",
            data_embedding_file="",
            data_full_embedding_file="",
            data_max_question_length=40,
            data_max_context_length=500,
            data_max_answer_length=30,
            data_max_subword_length=16,
            data_max_char_length=16,
            data_word_vocab_file="",
            data_word_vocab_size=50000,
            data_word_vocab_threshold=10,
            data_word_unk="<unk>",
            data_word_pad="<pad>",
            data_word_sos="<sos>",
            data_word_eos="<eos>",
            data_word_placeholder_enable=False,
            data_subword_vocab_file="",
            data_subword_vocab_size=20000,
            data_subword_vocab_threshold=20,
            data_subword_unk="***",
            data_subword_pad="###",
            data_subword_size=3,
            data_char_vocab_file="",
            data_char_vocab_size=1000,
            data_char_vocab_threshold=50,
            data_char_unk="*",
            data_char_pad="#",
            data_answer_type="",
            data_expand_multiple_answer=False,
            data_log_output_dir="",
            data_result_output_dir="",
            train_random_seed=100,
            train_enable_shuffle=True,
            train_batch_size=32,
            train_eval_batch_size=100,
            train_eval_metric=["exact", "f1"],
            train_eval_detail_type="full",
            train_decoding_sample_size=3,
            train_num_epoch=3,
            train_ckpt_output_dir="",
            train_summary_output_dir="",
            train_step_per_stat=10,
            train_step_per_ckpt=1000,
            train_step_per_eval=1000,
            train_clip_norm=5.0,
            train_ema_enable=True,
            train_ema_decay_rate=0.9999,
            train_optimizer_type="adam",
            train_optimizer_learning_rate=0.001,
            train_optimizer_decay_enable=True,
            train_optimizer_decay_mode="exponential_decay",
            train_optimizer_decay_rate=0.95,
            train_optimizer_decay_step=1000,
            train_optimizer_decay_start_step=10000,
            train_optimizer_momentum_beta=0.9,
            train_optimizer_rmsprop_beta=0.999,
            train_optimizer_rmsprop_epsilon=1e-8,
            train_optimizer_adadelta_rho=0.95,
            train_optimizer_adadelta_epsilon=1e-8,
            train_optimizer_adagrad_init_accumulator=0.1,
            train_optimizer_adam_beta_1=0.8,
            train_optimizer_adam_beta_2=0.999,
            train_optimizer_adam_epsilon=1e-07,
            model_type="qanet",
            model_scope="mrc",
            model_representation_word_embed_dim=300,
            model_representation_word_embed_pretrained=True,
            model_representation_word_feat_trainable=False,
            model_representation_word_feat_enable=True,
            model_representation_subword_embed_dim=200,
            model_representation_subword_unit_dim=100,
            model_representation_subword_window_size=[3, 5],
            model_representation_subword_hidden_activation="relu",
            model_representation_subword_dropout=0.05,
            model_representation_subword_pooling_type="max",
            model_representation_subword_feat_trainable=True,
            model_representation_subword_feat_enable=True,
            model_representation_char_embed_dim=200,
            model_representation_char_unit_dim=100,
            model_representation_char_window_size=[3, 5],
            model_representation_char_hidden_activation="relu",
            model_representation_char_dropout=0.05,
            model_representation_char_pooling_type="max",
            model_representation_char_feat_trainable=True,
            model_representation_char_feat_enable=True,
            model_representation_fusion_type="highway",
            model_representation_fusion_num_layer=2,
            model_representation_fusion_unit_dim=500,
            model_representation_fusion_hidden_activation="relu",
            model_representation_fusion_dropout=0.1,
            model_representation_fusion_trainable=True,
            model_understanding_question_num_layer=1,
            model_understanding_question_num_conv=4,
            model_understanding_question_num_head=8,
            model_understanding_question_unit_dim=128,
            model_understanding_question_window_size=[7],
            model_understanding_question_hidden_activation="relu",
            model_understanding_question_dropout=0.1,
            model_understanding_question_trainable=True,
            model_understanding_context_num_layer=1,
            model_understanding_context_num_conv=4,
            model_understanding_context_num_head=8,
            model_understanding_context_unit_dim=128,
            model_understanding_context_window_size=[7],
            model_understanding_context_hidden_activation="relu",
            model_understanding_context_dropout=0.1,
            model_understanding_context_trainable=True,
            model_understanding_enable_sharing=True,
            model_interaction_context2quesiton_attention_dim=128,
            model_interaction_context2quesiton_score_type="linear_plus",
            model_interaction_context2quesiton_trainable=True,
            model_interaction_context2quesiton_enable=True,
            model_interaction_quesiton2context_attention_dim=128,
            model_interaction_quesiton2context_score_type="linear_plus",
            model_interaction_quesiton2context_trainable=True,
            model_interaction_quesiton2context_enable=True,
            model_interaction_fusion_type="concate",
            model_interaction_fusion_num_layer=1,
            model_interaction_fusion_unit_dim=512,
            model_interaction_fusion_hidden_activation="relu",
            model_interaction_fusion_dropout=0.1,
            model_interaction_fusion_trainable=True,
            model_interaction_fusion_combo_enable=True,
            model_interaction_enable_sharing=True,
            model_modeling_answer_num_layer=7,
            model_modeling_answer_num_conv=2,
            model_modeling_answer_num_head=8,
            model_modeling_answer_unit_dim=128,
            model_modeling_answer_window_size=[5],
            model_modeling_answer_hidden_activation="relu",
            model_modeling_answer_dropout=0.1,
            model_modeling_answer_trainable=True,
            model_modeling_enable_sharing=True,
            model_output_answer_start_dropout=0.1,
            model_output_answer_start_trainable=True,
            model_output_answer_end_dropout=0.1,
            model_output_answer_end_trainable=True,
            device_num_gpus=1,
            device_default_gpu_id=0,
            device_log_device_placement=False,
            device_allow_soft_placement=False,
            device_allow_growth=False,
            device_per_process_gpu_memory_fraction=0.8
        )
    elif config_type == "rnet":
        hyperparams = tf.contrib.training.HParams(
            data_train_mrc_file="",
            data_train_mrc_file_type="",
            data_eval_mrc_file="",
            data_eval_mrc_file_type="",
            data_embedding_file="",
            data_full_embedding_file="",
            data_max_question_length=40,
            data_max_context_length=500,
            data_max_answer_length=30,
            data_max_subword_length=16,
            data_max_char_length=16,
            data_word_vocab_file="",
            data_word_vocab_size=50000,
            data_word_vocab_threshold=10,
            data_word_unk="<unk>",
            data_word_pad="<pad>",
            data_word_sos="<sos>",
            data_word_eos="<eos>",
            data_word_placeholder_enable=False,
            data_subword_vocab_file="",
            data_subword_vocab_size=20000,
            data_subword_vocab_threshold=20,
            data_subword_unk="***",
            data_subword_pad="###",
            data_subword_size=3,
            data_char_vocab_file="",
            data_char_vocab_size=1000,
            data_char_vocab_threshold=50,
            data_char_unk="*",
            data_char_pad="#",
            data_answer_type="",
            data_expand_multiple_answer=False,
            data_log_output_dir="",
            data_result_output_dir="",
            train_random_seed=100,
            train_enable_shuffle=True,
            train_batch_size=60,
            train_eval_batch_size=100,
            train_eval_metric=["exact", "f1"],
            train_eval_detail_type="full",
            train_decoding_sample_size=3,
            train_num_epoch=3,
            train_ckpt_output_dir="",
            train_summary_output_dir="",
            train_step_per_stat=10,
            train_step_per_ckpt=1000,
            train_step_per_eval=1000,
            train_clip_norm=5.0,
            train_ema_enable=True,
            train_ema_decay_rate=0.999,
            train_optimizer_type="adam",
            train_optimizer_learning_rate=0.001,
            train_optimizer_decay_enable=False,
            train_optimizer_decay_mode="exponential_decay",
            train_optimizer_decay_rate=0.95,
            train_optimizer_decay_step=1000,
            train_optimizer_decay_start_step=10000,
            train_optimizer_momentum_beta=0.9,
            train_optimizer_rmsprop_beta=0.999,
            train_optimizer_rmsprop_epsilon=1e-8,
            train_optimizer_adadelta_rho=0.95,
            train_optimizer_adadelta_epsilon=1e-8,
            train_optimizer_adagrad_init_accumulator=0.1,
            train_optimizer_adam_beta_1=0.9,
            train_optimizer_adam_beta_2=0.999,
            train_optimizer_adam_epsilon=1e-08,
            model_type="rnet",
            model_scope="mrc",
            model_representation_word_embed_dim=300,
            model_representation_word_embed_pretrained=True,
            model_representation_word_feat_trainable=False,
            model_representation_word_feat_enable=True,
            model_representation_subword_embed_dim=20,
            model_representation_subword_unit_dim=100,
            model_representation_subword_window_size=[3, 5],
            model_representation_subword_hidden_activation="relu",
            model_representation_subword_dropout=0.2,
            model_representation_subword_pooling_type="max",
            model_representation_subword_feat_trainable=True,
            model_representation_subword_feat_enable=True,
            model_representation_char_embed_dim=20,
            model_representation_char_unit_dim=100,
            model_representation_char_window_size=[3, 5],
            model_representation_char_hidden_activation="relu",
            model_representation_char_dropout=0.2,
            model_representation_char_pooling_type="max",
            model_representation_char_feat_trainable=True,
            model_representation_char_feat_enable=True,
            model_representation_fusion_type="highway",
            model_representation_fusion_num_layer=2,
            model_representation_fusion_unit_dim=150,
            model_representation_fusion_hidden_activation="relu",
            model_representation_fusion_dropout=0.2,
            model_representation_fusion_trainable=True,
            model_understanding_question_num_layer=3,
            model_understanding_question_unit_dim=75,
            model_understanding_question_cell_type="gru",
            model_understanding_question_hidden_activation="tanh",
            model_understanding_question_dropout=0.2,
            model_understanding_question_forget_bias=False,
            model_understanding_question_residual_connect=False,
            model_understanding_question_trainable=True,
            model_understanding_context_num_layer=3,
            model_understanding_context_unit_dim=75,
            model_understanding_context_cell_type="gru",
            model_understanding_context_hidden_activation="tanh",
            model_understanding_context_dropout=0.2,
            model_understanding_context_forget_bias=False,
            model_understanding_context_residual_connect=False,
            model_understanding_context_trainable=True,
            model_understanding_enable_sharing=False,
            model_interaction_context2quesiton_num_layer=1,
            model_interaction_context2quesiton_unit_dim=75,
            model_interaction_context2quesiton_hidden_activation="relu",
            model_interaction_context2quesiton_dropout=0.2,
            model_interaction_context2quesiton_attention_dim=75,
            model_interaction_context2quesiton_score_type="linear",
            model_interaction_context2quesiton_trainable=True,
            model_interaction_context2quesiton_enable=True,
            model_interaction_fusion_type="concate",
            model_interaction_fusion_num_layer=1,
            model_interaction_fusion_unit_dim=150,
            model_interaction_fusion_hidden_activation="relu",
            model_interaction_fusion_dropout=0.2,
            model_interaction_fusion_trainable=True,
            model_modeling_answer_num_layer=1,
            model_modeling_answer_unit_dim=75,
            model_modeling_answer_cell_type="gru",
            model_modeling_answer_hidden_activation="tanh",
            model_modeling_answer_dropout=0.2,
            model_modeling_answer_forget_bias=False,
            model_modeling_answer_residual_connect=False,
            model_modeling_answer_attention_dim=75,
            model_modeling_answer_score_type="linear",
            model_modeling_answer_trainable=True,
            model_modeling_fusion_type="concate",
            model_modeling_fusion_num_layer=1,
            model_modeling_fusion_unit_dim=150,
            model_modeling_fusion_hidden_activation="relu",
            model_modeling_fusion_dropout=0.2,
            model_modeling_fusion_trainable=True,
            model_output_answer_start_num_layer=1,
            model_output_answer_start_unit_dim=75,
            model_output_answer_start_cell_type="gru",
            model_output_answer_start_hidden_activation="tanh",
            model_output_answer_start_dropout=0.2,
            model_output_answer_start_forget_bias=False,
            model_output_answer_start_residual_connect=False,
            model_output_answer_start_trainable=True,
            model_output_answer_end_num_layer=1,
            model_output_answer_end_unit_dim=75,
            model_output_answer_end_cell_type="gru",
            model_output_answer_end_hidden_activation="tanh",
            model_output_answer_end_dropout=0.2,
            model_output_answer_end_forget_bias=False,
            model_output_answer_end_residual_connect=False,
            model_output_answer_end_trainable=True,
            device_num_gpus=1,
            device_default_gpu_id=0,
            device_log_device_placement=False,
            device_allow_soft_placement=False,
            device_allow_growth=False,
            device_per_process_gpu_memory_fraction=0.8
        )
    else:
        raise ValueError("unsupported config type {0}".format(config_type))
    
    return hyperparams

def load_hyperparams(config_file):
    """load hyperparameters from config file"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams_dict = json.load(file)
            hyperparams = create_default_hyperparams(hyperparams_dict["model_type"])
            hyperparams.set_from_map(hyperparams_dict)
            
            return hyperparams
    else:
        raise FileNotFoundError("config file not found")

def generate_search_lookup(search,
                           search_lookup=None):
    search_lookup = search_lookup if search_lookup else {}
    search_type = search["stype"]
    data_type = search["dtype"]
    
    if search_type == "uniform":
        range_start = search["range"][0]
        range_end = search["range"][1]
        if data_type == "int":
            search_sample = np.random.randint(range_start, range_end)
        elif data_type == "float":
            search_sample = (range_end - range_start) * np.random.random_sample() + range_start
        else:
            raise ValueError("unsupported data type {0}".format(data_type))
    elif search_type == "log":
        range_start = math.log(search["range"][0], 10)
        range_end = math.log(search["range"][1], 10)
        if data_type == "float":
            search_sample = math.pow(10, (range_end - range_start) * np.random.random_sample() + range_start)
        else:
            raise ValueError("unsupported data type {0}".format(data_type))
    elif search_type == "discrete":
        search_set = search["set"]
        search_index = np.random.choice(len(search_set))
        search_sample = search_set[search_index]
    elif search_type == "lookup":
        search_key = search["key"]
        if search_key in search_lookup:
            search_sample = search_lookup[search_key]
        else:
            raise ValueError("search key {0} doesn't exist in look-up table".format(search_key))
    else:
        raise ValueError("unsupported search type {0}".format(search_type))
    
    data_scale = search["scale"] if "scale" in search else 1.0
    data_shift = search["shift"] if "shift" in search else 0.0
    
    if data_type == "int":
        search_sample = int(data_scale * search_sample + data_shift)
    elif data_type == "float":
        search_sample = float(data_scale * search_sample + data_shift)
    elif data_type == "string":
        search_sample = str(search_sample)
    elif data_type == "boolean":
        search_sample = bool(search_sample)
    elif data_type == "list":
        search_sample = list(search_sample)
    else:
        raise ValueError("unsupported data type {0}".format(data_type))
    
    return search_sample

def search_hyperparams(hyperparams,
                       config_file,
                       num_group,
                       random_seed):
    """search hyperparameters based on search config"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams_group = []
            np.random.seed(random_seed)
            search_setting = json.load(file)
            hyperparams_search_setting = search_setting["hyperparams"]
            variables_search_setting = search_setting["variables"]
            for i in range(num_group):
                variables_search_lookup = {}
                for key in variables_search_setting.keys():
                    variables_search = variables_search_setting[key]
                    variables_search_lookup[key] = generate_search_lookup(variables_search)
                hyperparams_search_lookup = {}
                for key in hyperparams_search_setting.keys():
                    hyperparams_search = hyperparams_search_setting[key]
                    hyperparams_search_lookup[key] = generate_search_lookup(hyperparams_search, variables_search_lookup)
                
                hyperparams_sample = tf.contrib.training.HParams(hyperparams.to_proto())
                hyperparams_sample.set_from_map(hyperparams_search_lookup)
                hyperparams_group.append(hyperparams_sample)
            
            return hyperparams_group
    else:
        raise FileNotFoundError("config file not found")

def create_hyperparams_file(hyperparams_group, config_dir):
    """create config files from groups of hyperparameters"""
    if not tf.gfile.Exists(config_dir):
        tf.gfile.MakeDirs(config_dir)
    
    for i in range(len(hyperparams_group)):
        config_file = os.path.join(config_dir, "config_hyperparams_{0}.json".format(i))
        with codecs.getwriter("utf-8")(tf.gfile.GFile(config_file, "w")) as file:
            hyperparams_json = json.dumps(hyperparams_group[i].values(), indent=4)
            file.write(hyperparams_json)
