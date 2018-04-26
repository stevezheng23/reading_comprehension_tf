import numpy as np
import tensorflow as tf

__all__ = ["check_tensorflow_version"]

def check_tensorflow_version():
    """check tensorflow version in current environment"""
    min_tf_version = "1.4.0"
    curr_tf_version = tf.__version__
    if curr_tf_version < min_tf_version:
        raise EnvironmentError("tensorflow version must be >= {0}".format(min_tf_version))
    return curr_tf_version