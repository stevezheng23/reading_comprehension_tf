import codecs
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["TrainLogger"]

class TrainLogger(object):
    """train logger"""
    def __init__(self,
                 output_dir):
        """initialize train logger"""
        self.loss = 0.0
        self.learning_rate = 0.0
        self.global_step = 0
        self.epoch = 0
        self.step_in_epoch = 0
        self.train_time = 0.0
        self.sample_size = 0
        self.prev_check_loss = 0.0
        self.prev_check_train_time = 0.0
        self.prev_check_sample_size = 0
        
        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)
        self.log_file = os.path.join(output_dir, "train_{0}.log".format(time.time()))
        self.log_writer = codecs.getwriter("utf-8")(tf.gfile.GFile(self.log_file, mode="a"))
    
    def update(self,
               train_result,
               epoch,
               step_in_epoch,
               time_per_step):
        """update train logger based on train result"""
        self.loss += train_result.loss * train_result.batch_size
        self.learning_rate = train_result.learning_rate
        self.global_step = train_result.global_step
        self.epoch = epoch
        self.step_in_epoch = step_in_epoch
        self.train_time += time_per_step
        self.sample_size += train_result.batch_size
    
    def check(self):
        """check train statistic"""
        loss_delta = self.loss - self.prev_check_loss
        train_time_delta = self.train_time - self.prev_check_train_time
        sample_size_delta = self.sample_size - self.prev_check_sample_size
        
        if self.sample_size <= 0:
            raise ValueError("current sample size is less than or equal to 0")
        
        if sample_size_delta <= 0:
            return
        
        avg_loss = loss_delta / sample_size_delta
        curr_loss = self.loss / self.sample_size
        
        log_line = "epoch={0}, step={1}, global step={2}, train time={3} avg. loss={4}, curr loss={5}".format(
            self.epoch, self.step_in_epoch, self.global_step, train_time_delta, avg_loss, curr_loss).encode('utf-8')
        self.log_writer.write("{0}\r\n".format(log_line))
        print(log_line)
        
        self.prev_check_loss = self.loss
        self.prev_check_train_time = self.train_time
        self.prev_check_sample_size = self.sample_size      
