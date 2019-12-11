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

from transformer import Transformer
from code_data_reader import CodeDataReader
from metrics import MetricsTracker
from log_saver import LogSaver
from code_comment_aligner import log_and_print
from model import CodeTransformerModel, BaselineModel

import tensorflow as tf


config = yaml.safe_load(open("config.yml"))


def create_checkpoint(config, model, optimizer):
    checkpoint_path = config["training"]["checkpoint"]
    ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)
    return ckpt, ckpt_manager


def loss_function(targets, predictions):
    return tf.losses.binary_crossentropy(targets, predictions, label_smoothing=.01)


def get_learning_rate():
    return tf.constant(config["training"]["lr"])


def train(model, data):
    optimizer = tf.optimizers.Adam(get_learning_rate)
    log = LogSaver(config["training"]["log_dir"], config["training"]
                   ["model"], config["training"]["input_type"], "code")

    ckpt, ckpt_manager = create_checkpoint(config, model, optimizer)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Restored latest checkpoint')

    total_batches = 0
    is_first = True
    for epoch in range(config["training"]["num_epochs"]):
        log_and_print("Epoch: {}".format(epoch + 1))
        metrics = MetricsTracker()
        mbs = 0
        s = time.perf_counter()
        for batch in data.batcher(mode="training"):
            mbs += 1
            total_batches += 1

            # Run through one batch to init variables
            if is_first:
                model(*batch[:-1])
                is_first = False
                log_and_print("Model initialized, training {:,} parameters".format(
                    np.sum([np.prod(v.shape) for v in model.trainable_variables])))

            loss, preds = train_step(batch, model, optimizer)

            metrics.add_observation(batch[-1], preds, loss)
            log.log_train(loss, metrics.get_acc(),
                          metrics.get_bce(), total_batches)

            if mbs % config["training"]["print_freq"] == 0:
                lr = optimizer.get_config()['learning_rate'].numpy()
                log_and_print("MB: {0}, lr: {1:.1e}: samples: {2:,}, entropy: {3}, acc: {4} loss: {5:.4f}".format(
                    mbs, lr, *metrics.get_stats(), loss))
                metrics.flush()

        print("Time per epoch {:0.2f} seconds.".format(
            time.perf_counter() - s))
        print('Saving checkpoint for epoch {} at {}'.format(
            epoch+1, ckpt_manager.save()))
        # Run a validation pass at the end of every epoch
        log_and_print("Validation: samples: {0}, entropy: {1}, accs: {2}".format(
            *eval(model, data)))
        log.log_valid(metrics.get_acc(), metrics.get_bce(), epoch + 1)

    log_and_print(
        "Test: samples: {0}, entropy: {1}, accs: {2}".format(*eval(model, data, validate=False)))


def train_step(batch, model, optimizer):
    # Compute loss in scope of gradient-tape (can also use implicit gradients)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        preds = model(*batch[:-1])
        loss = loss_function(batch[-1], preds)
    # Collect gradients, clip and apply
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 0.25)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, preds


def eval(model, data, validate=True):
    mbs = 0
    metrics = MetricsTracker()
    for batch in data.batcher(mode="valid" if validate else "test"):
        mbs += 1
        preds = model(*batch[:-1], training=False)
        metrics.add_observation(
            batch[-1], preds, loss_function(batch[-1], preds))
    return metrics.get_stats()


def test(model, data):
    optimizer = tf.optimizers.Adam(get_learning_rate)
    ckpt, ckpt_manager = create_checkpoint(config, model, optimizer)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Restored latest checkpoint')

    print("Test: samples: {0}, entropy: {1}, accs: {2}".format(
        *eval(model, data, validate=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="Path to training data")
    ap.add_argument("-m", "--mode", required=False,
                    help="Mode to run 'train' to train and 'test' to test", default="train")
    ap.add_argument("-t", "--test", required=False,
                    help="Path to test file")
    ap.add_argument("-v", "--vocab", required=False,
                    help="Path to vocabulary files")
    ap.add_argument("-r", "--restore", type=bool,
                    required=False, default=False)
    args = ap.parse_args()

    data = CodeDataReader(
        config["data"], args.data, test_file=args.test, vocab_file=args.vocab)
    model = CodeTransformerModel(
        config["transformer"], data.vocabulary.vocab_size)
    # model = BaselineModel(config['baseline'], data.vocabulary.vocab_size)

    if args.mode == "train":
        train(model, data, args.restore)
    if args.mode == "test":
        test(model, data)


main()
