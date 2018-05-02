import numpy as np
import tensorflow as tf

__all__ = ["SummaryWriter"]

class SummaryWriter(object):
    """summary writer"""    
    def __init__(self,
                 graph,
                 output_dir):
        """initialize summary writer"""       
        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)
        self.summary_writer = tf.summary.FileWriter(output_dir, graph)
    
    def add_summary(self,
                    summary,
                    global_step):
        """add new summary"""
        self.summary_writer.add_summary(summary, global_step)
    
    def add_value_summary(self,
                          summary_tag,
                          summary_value,
                          global_step):
        """add new value summary"""
        summary = tf.Summary(value=[tf.Summary.Value(tag=summary_tag, simple_value=summary_value)])
        self.summary_writer.add_summary(summary, global_step)

    def close_writer(self):
        """close summary writer"""
        self.summary_writer.close()
    
    def reopen_writer(self):
        """re-open summary writer"""
        self.summary_writer.reopen()
