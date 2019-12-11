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

    def reshape_masks(self, masks):
        return tf.reshape(masks, [masks.shape[0], 1, 1, masks.shape[1]])

    def build_key_masks(self, self_masks, masks):
        return self_masks * tf.reshape(masks, [masks.shape[0], 1, masks.shape[1], 1])

    def call(self, b_code_indices, b_code_masks, a_code_indices, a_code_masks, comment_indices, comment_masks, training=True):
        # setting up masks
        b_code_self_masks = self.reshape_masks(b_code_masks)
        a_code_self_masks = self.reshape_masks(a_code_masks)
        comment_self_masks = self.reshape_masks(comment_masks)

        a_comment_key_masks = self.build_key_masks(
            comment_self_masks, a_code_masks)
        b_comment_key_masks = self.build_key_masks(
            comment_self_masks, b_code_masks)

        # comment
        comment_states = self.transformer(
            comment_indices, masks=comment_self_masks, training=training)

        # before code
        b_states = self.transformer(b_code_indices, masks=b_code_self_masks, key_states=comment_states,
                                    key_masks=b_comment_key_masks, training=training)
        # after code
        a_states = self.transformer(a_code_indices, masks=a_code_self_masks, key_states=comment_states,
                                    key_masks=a_comment_key_masks, training=training)

        b_states = tf.reduce_max(b_states, axis=1)
        a_states = tf.reduce_max(a_states, axis=1)
        comment_states = tf.reduce_max(comment_states, axis=1)

        states = self.interaction(comment_states, b_states, a_states)

        # Max-pool states and project to classify
        states = self.fc2(self.fc1(states))
        preds = self.classify(states)
        preds = tf.squeeze(preds, -1)
        return preds

    def interaction(self, comment_states, b_states, a_states):
        b_com_interaction = tf.square(comment_states - b_states)
        a_com_interaction = tf.square(comment_states - a_states)
        x5 = tf.square(comment_states) - tf.square(b_states)
        x6 = tf.square(comment_states) - tf.square(a_states)
        return tf.concat(
            [b_states, comment_states, a_states, b_com_interaction, a_com_interaction, x5, x6], axis=-1)


class BaselineModel(tf.keras.layers.Layer):

    def __init__(self, model_config, token_vocab_dim):
        super(BaselineModel, self).__init__()
        random_init = tf.random_normal_initializer(
            stddev=model_config["hidden_dim"] ** -0.5)
        self.embed = tf.keras.layers.Embedding(
            token_vocab_dim, model_config["embed_dim"])

        self.lstm_layers = [tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(model_config["hidden_dim"], dropout=model_config["dropout_rate"],
                                recurrent_dropout=model_config["dropout_rate"], return_sequences=True))
                            for _ in range(model_config["num_layers"])]
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(model_config["hidden_dim"], dropout=model_config["dropout_rate"],
                                recurrent_dropout=model_config["dropout_rate"]))

        self.attention = tf.keras.layers.Attention()
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(model_config["dropout_rate"])
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

    def rnn_layers(self, states):
        for ix in range(len(self.lstm_layers)):
            states = self.lstm_layers[ix](states)
        return self.lstm(states)
        # return states

    def rnn_layers(self, states1, states2):
        for ix in range(len(self.lstm_layers)):
            states1 = self.lstm_layers[ix](states1)
            states2 = self.lstm_layers[ix](states2)
            attn_out = self.attention([states1, states2])
        return self.lstm(states1), self.lstm(states2), attn_out

    def call(self, b_code_indices, b_code_masks, a_code_indices, a_code_masks, comment_indices, comment_masks, training=True):
        comment_em = self.embed(comment_indices)
        b_code_em = self.embed(b_code_indices)
        a_code_em = self.embed(a_code_indices)

        b_code_states, b_com_states, attn_out1 = self.rnn_layers(
            b_code_em, comment_em)  # [B, 2 * H]
        a_code_states, a_com_states, attn_out2 = self.rnn_layers(
            a_code_em, comment_em)  # [B, 2 * H]

        attn_out1 = tf.keras.layers.GlobalAveragePooling1D()(attn_out1)
        attn_out2 = tf.keras.layers.GlobalAveragePooling1D()(attn_out2)

        states = tf.concat(
            [b_code_states, b_com_states, a_code_states, a_com_states, attn_out1, attn_out2], axis=-1)

        states = self.fc1(states)
        states = self.dropout1(states, training=training)
        states = self.batch_norm1(states)
        preds = self.classify(states)  # [B, 1]
        preds = tf.squeeze(preds, -1)  # [B]
        return preds
