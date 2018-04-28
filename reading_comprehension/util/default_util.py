import math

import numpy as np
import tensorflow as tf

__all__ = ["check_tensorflow_version", "safe_exp", "get_config_proto", "get_device_spec"]

def check_tensorflow_version():
    """check tensorflow version in current environment"""
    min_tf_version = "1.4.0"
    curr_tf_version = tf.__version__
    if curr_tf_version < min_tf_version:
        raise EnvironmentError("tensorflow version must be >= {0}".format(min_tf_version))
    return curr_tf_version

def safe_exp(value):
    """handle overflow exception for math.exp"""
    try:
        res = math.exp(value)
    except OverflowError:
        res = float("inf")    
    return res

def get_config_proto(log_device_placement,
                     allow_soft_placement,
                     allow_growth,
                     per_process_gpu_memory_fraction):
    """get config proto for device setting"""
    config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = allow_growth
    config_proto.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    
    return config_proto

def get_device_spec(device_id, num_gpus):
    """get device specification"""
    if num_gpus == 0:
        device_spec = "/device:CPU:0"
    else:
        device_spec = "/device:GPU:{0}".format(device_id % num_gpus)
    
    return device_spec
