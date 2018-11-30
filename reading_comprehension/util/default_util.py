import numpy as np
import tensorflow as tf

__all__ = ["EPSILON", "MAX_INT", "MIN_FLOAT", "check_tensorflow_version", "safe_exp", "get_config_proto", "get_device_spec"]

EPSILON = 1e-30
MAX_INT = 2147483647
MIN_FLOAT = -1e30

def check_tensorflow_version():
    """check tensorflow version in current environment"""
    min_tf_version = "1.12.0"
    curr_tf_version = tf.__version__
    if curr_tf_version < min_tf_version:
        raise EnvironmentError("tensorflow version must be >= {0}".format(min_tf_version))
    return curr_tf_version

def safe_exp(value):
    """handle overflow exception for exp"""
    try:
        res = np.exp(value)
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
