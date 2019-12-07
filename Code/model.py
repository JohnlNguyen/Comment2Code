import sys
import os
import math
import random
import yaml
import argparse
import time
import pprint
import pickle
import numpy as np

from pdb import set_trace
from transformer import Transformer
from code_data_reader import CodeDataReader
from metrics import MetricsTracker
from log_saver import LogSaver
from code_comment_aligner import log_and_print
import tensorflow as tf


class CodeTransformerModel(tf.keras.Model):
    def __init__(self, model_config, token_vocab_dim):
        super(CodeTransformerModel, self).__init__()
        self.transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim,
                                       model_config[
            "attention_dim"], model_config["num_layers"], model_config["ff_dim"],
            model_config["num_heads"], model_config["dropout_rate"])

        self.fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.fc2 = tf.keras.layers.Dense(512, activation="relu")
        self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, b_code_indices, b_code_masks, a_code_indices, a_code_masks, comment_indices, comment_masks):
        # setting up masks
        b_code_self_masks = tf.reshape(
            b_code_masks, [b_code_masks.shape[0], 1, 1, b_code_masks.shape[1]])
        a_code_self_masks = tf.reshape(
            a_code_masks, [a_code_masks.shape[0], 1, 1, a_code_masks.shape[1]])
        comment_self_masks = tf.reshape(
            comment_masks, [comment_masks.shape[0], 1, 1, comment_masks.shape[1]])

        a_comment_key_masks = comment_self_masks * tf.reshape(a_code_masks,
                                                              [a_code_masks.shape[0], 1, a_code_masks.shape[1], 1])
        b_comment_key_masks = comment_self_masks * tf.reshape(b_code_masks,
                                                              [b_code_masks.shape[0], 1, b_code_masks.shape[1], 1])

        b_code_key_masks = b_code_self_masks * \
            tf.reshape(comment_masks, [
                       comment_masks.shape[0], 1, comment_masks.shape[1], 1])

        a_code_key_masks = a_code_self_masks * \
            tf.reshape(comment_masks, [
                       comment_masks.shape[0], 1, comment_masks.shape[1], 1])

        # comment
        comment_states = self.transformer(
            comment_indices, masks=comment_self_masks)

        # before code
        b_states = self.transformer(b_code_indices, masks=b_code_self_masks, key_states=comment_states,
                                    key_masks=b_comment_key_masks)
        # after code
        a_states = self.transformer(a_code_indices, masks=a_code_self_masks, key_states=comment_states,
                                    key_masks=a_comment_key_masks)

        comment_b_states = self.transformer(
            comment_indices, masks=comment_self_masks, key_states=b_states, key_masks=b_code_key_masks)
        comment_a_states = self.transformer(
            comment_indices, masks=comment_self_masks, key_states=a_states, key_masks=a_code_key_masks)

        b_states = tf.reduce_max(b_states, axis=1)
        a_states = tf.reduce_max(a_states, axis=1)
        comment_states = tf.reduce_max(comment_states, axis=1)
        comment_b_states = tf.reduce_max(comment_b_states, axis=1)
        comment_a_states = tf.reduce_max(comment_a_states, axis=1)

        states = self.interaction(comment_states, a_states, b_states)
        states = tf.concat([states, comment_b_states, comment_a_states], 1)

        # Max-pool states and project to classify
        states = self.fc2(self.fc1(states))
        preds = self.classify(states)
        preds = tf.squeeze(preds, -1)
        return preds

    def interaction(self, comment, b_code, a_code):
        b_minus = tf.square(b_code - comment)
        a_minus = tf.square(a_code - comment)
        b_plus = tf.square(b_code + comment)
        a_plus = tf.square(a_code + comment)
        return tf.concat([comment, b_code, a_code, b_minus, a_minus, b_plus, a_plus], -1)
