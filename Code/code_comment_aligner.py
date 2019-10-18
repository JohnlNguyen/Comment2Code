import sys
import os
import math
import random
import yaml
import argparse
import time

import numpy as np
import tensorflow as tf

from transformer import Transformer
from metrics import MetricsTracker
from data_reader import DataReader
import util

random.seed(41)
config = yaml.safe_load(open("config.yml"))

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    # Extract arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="Path to training data")
    ap.add_argument("-v", "--vocabulary", required=False, help="Path to vocabulary files")
    args = ap.parse_args()
    print("Using configuration:", config)
    data = DataReader(config["data"], config["vocabulary"], args.data, vocab_file=args.vocabulary)
    if config["training"]["model"] == "transformer":
        model = TransformerModel(config["transformer"], data.vocabulary.vocab_dim)
    elif config["training"]["model"] == "baseline" or config["training"]["model"] == "rnn":
        model = BaselineModel(config["baseline"], data.vocabulary.vocab_dim)
    train(model, data)


def train(model, data):
    # Declare the learning rate as a function to include it in the saved state
    def get_learning_rate():
        if "transformer" not in config["training"]["model"]:
            return tf.constant(config["training"]["lr"])
        return tf.constant(4e-4)
    optimizer = tf.optimizers.Adam(get_learning_rate)

    total_batches = 0
    is_first = True
    for epoch in range(config["training"]["num_epochs"]):
        print("Epoch:", epoch + 1)
        metrics = MetricsTracker()
        mbs = 0
        for batch in data.batcher(mode="training"):
            mbs += 1
            total_batches += 1

            # Run through one batch to init variables
            if is_first:
                model(*batch[:-1])
                is_first = False
                print("Model initialized, training {:,} parameters".format(
                    np.sum([np.prod(v.shape) for v in model.trainable_variables])))

            # Compute loss in scope of gradient-tape (can also use implicit gradients)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_variables)
                preds = model(*batch[:-1])
                loss = get_loss(batch[-1], preds)

            # Collect gradients, clip and apply
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.25)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            metrics.add_observation(batch[-1], preds, loss)
            if mbs % config["training"]["print_freq"] == 0:
                lr = optimizer.get_config()['learning_rate'].numpy()
                print("MB: {0}, lr: {1:.4f}: samples: {2:,}, entropy: {3}, acc: {4}".format(
                    mbs, lr, *metrics.get_stats()))
                metrics.flush()
        # Run a validation pass at the end of every epoch
        print("Validation: samples: {0}, entropy: {1}, accs: {2}".format(*eval(model, data)))
    print("Test: samples: {0}, entropy: {1}, accs: {2}".format(*eval(model, data, validate=False)))


def eval(model, data, validate=True):
    mbs = 0
    metrics = MetricsTracker()
    for batch in data.batcher(mode="valid" if validate else "test"):
        mbs += 1
        preds = model(*batch[:-1])
        metrics.add_observation(batch[-1], preds, get_loss(batch[-1], preds))
    return metrics.get_stats()


def get_loss(targets, predictions):
    return tf.reduce_mean(-tf.math.log(1e-6 + predictions) * targets + -tf.math.log(1e-6 + 1 - predictions) * (1 - targets))


class TransformerModel(tf.keras.layers.Layer):
	def __init__(self, model_config, token_vocab_dim):
    		super(TransformerModel, self).__init__()
		self.code_transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim, model_config["attention_dim"], model_config["num_layers"], model_config["ff_dim"], model_config["num_heads"], model_config["dropout_rate"])
		self.comment_transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim, model_config["attention_dim"], model_config["num_layers"], model_config["ff_dim"], model_config["num_heads"], model_config["dropout_rate"])
		self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

	def call(self, comment_indices, comment_masks, code_indices, code_masks):
		# Set up masks for our two transformers
		comment_self_masks = tf.reshape(comment_masks, [comment_masks.shape[0], 1, 1, comment_masks.shape[1]])
		code_self_masks = tf.reshape(code_masks, [code_masks.shape[0], 1, 1, code_masks.shape[1]])
		code_key_masks = code_self_masks * tf.reshape(comment_masks, [comment_masks.shape[0], 1, comment_masks.shape[1], 1])

		# Compute code self-attention states
		code_states = self.code_transformer(code_indices, masks=code_self_masks)

		# Compute comment self+code-attention states
		states = self.comment_transformer(comment_indices, masks=comment_self_masks, key_states=code_states, key_masks=code_key_masks)

		# Max-pool states and project to classify
		states = tf.reduce_max(states, 1)
		preds = self.classify(states)
		preds = tf.squeeze(preds, -1)
		return preds

class BaselineModel(tf.keras.layers.Layer):

    def __init__(self, model_config, token_vocab_dim):
        super(BaselineModel, self).__init__()
        random_init = tf.random_normal_initializer(stddev=model_config["hidden_dim"] ** -0.5)
        self.embed = tf.Variable(random_init(
            [token_vocab_dim, model_config["embed_dim"]]), dtype=tf.float32)
        self.rnns_fwd_code = [tf.keras.layers.GRU(
            model_config["embed_dim"] // 2, return_sequences=True) for _ in range(2)]
        self.rnns_bwd_code = [tf.keras.layers.GRU(
            model_config["hidden_dim"] // 2, return_sequences=True, go_backwards=True) for _ in range(2)]
        self.rnns_fwd_comment = [tf.keras.layers.GRU(
            model_config["embed_dim"] // 2, return_sequences=True) for _ in range(2)]
        self.rnns_bwd_comment = [tf.keras.layers.GRU(
            model_config["hidden_dim"] // 2, return_sequences=True, go_backwards=True) for _ in range(2)]
        self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, comment_indices, comment_masks, code_indices, code_masks):
        comment_masks = tf.expand_dims(comment_masks, -1)
        code_masks = tf.expand_dims(code_masks, -1)

        def run_base(indices, masks, rnns_fwd, rnns_bwd):
            embs = tf.nn.embedding_lookup(self.embed, indices)
            embs *= masks
            states = embs
            for ix in range(len(rnns_fwd)):
                fwd = rnns_fwd[ix](states)
                bwd = rnns_bwd[ix](states)
                states = tf.concat([fwd, bwd], axis=-1)
                states *= masks
            return states
        comment_states = run_base(comment_indices, comment_masks,
                                  self.rnns_fwd_comment, self.rnns_bwd_comment)
        code_states = run_base(code_indices, code_masks, self.rnns_fwd_code, self.rnns_bwd_code)
        states = tf.concat([tf.reduce_max(code_states, 1), tf.reduce_max(comment_states, 1)], -1)
        return self.classify(states)

if __name__ == '__main__':
    main()
