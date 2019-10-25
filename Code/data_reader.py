import numpy as np

import os
import sys
import re
import random
import json
random.seed(42)

import yaml
config = yaml.safe_load(open("config.yml"))

from vocabulary import VocabularyBuilder
from pdb import set_trace

import tensorflow as tf

# import jsonlines
import csv


class DataReader(object):

    def __init__(self, data_config, vocab_config, data_root, vocab_file):
        self.config = data_config
        self.train_data, self.valid_data, self.test_data = self.read(data_root)
        print("%d lines" % len(self.train_data))
        self.get_vocab(vocab_config, vocab_file)

        # Limit held-out data size
        if sum(len(l) for l in self.valid_data) > 1000000:
            random.shuffle(self.valid_data)
            self.valid_data = self.valid_data[:250]

    def get_vocab(self, vocab_config, vocab_file):
        if vocab_file != None:
            self.vocabulary = VocabularyBuilder(vocab_config, vocab_path=vocab_file)
        else:
            self.vocabulary = VocabularyBuilder(
                vocab_config, file_contents=list(self.train_tokens()))

    def train_tokens(self):
        for l in self.train_data:
            yield l["before_comment"].replace("\n", "\\n")
            yield "\\n".join(l["before_code"]).replace("\n", "\\n")
            yield l["after_comment"].replace("\n", "\\n")
            yield "\\n".join(l["after_code"]).replace("\n", "\\n")

    def read(self, data_root):
        if os.path.isdir(data_root):
            data = []
            for file in os.listdir(data_root):
                with open(os.path.join(data_root, file), encoding='utf-8', errors='ignore') as f:
                    data.append(json.load(f))
            train_data = [k for l in data[:int(0.9 * len(data))] for k in l]
            valid_data = [k for l in data[int(0.9 * len(data)):int(0.95 * len(data))] for k in l]
            test_data = [k for l in data[int(0.95 * len(data))] for k in l]
            return train_data, valid_data, test_data
        else:
            """normal"""
            with open(data_root, encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            """dcs"""
            # data = []
            # with jsonlines.open(data_root) as f:
            #     data = [line for line in f]
            """quora"""
            # data = []
            # with open(data_root, newline='') as csv_file:
            #     reader = csv.DictReader(csv_file, delimiter=',')
            #     data = [row for row in reader]

            # train_data = data[:int(0.9 * len(data))]
            # valid_data = data[int(0.9 * len(data)):int(0.95 * len(data))]
            # test_data = data[int(0.95 * len(data))]

            train_data = data[:int(0.95 * len(data))]
            valid_data = data[int(0.95 * len(data)):]
            test_data = []
            return train_data, valid_data, test_data

    def batcher(self, mode="training"):
        ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
            tf.int32, tf.float32, tf.int32, tf.float32, tf.float32), args=(mode,))
        ds = ds.prefetch(1)
        return ds

    def stats(self):
        print("Num OOV {}".format(self.vocabulary.num_oov))

    def batch_generator(self, mode="training"):
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if mode == "training":
            batch_data = self.train_data
            random.shuffle(batch_data)
        elif mode == "valid":
            batch_data = self.valid_data
        else:
            batch_data = self.test_data

        sample_len = lambda l: len(l[0]) + len(l[1])

        def make_batch(buffer):
            pivot = sample_len(random.choice(buffer))
            buffer = sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))
            batch = [[], [], []]
            max_seq_len = 0
            for ix, seq in enumerate(buffer):
                max_seq_len = max(max_seq_len, sample_len(seq))
                if max_seq_len * (len(batch[0]) + 1) > config['data']['max_batch_size']:
                    break

                batch[0].append([self.vocabulary.vocab_key(s) for s in seq[0]])
                batch[1].append([self.vocabulary.vocab_key(s) for s in seq[1]])
                batch[2].append(seq[2])
            comment_indices = tf.ragged.constant(batch[0], dtype="int32").to_tensor()
            comment_masks = tf.sequence_mask([len(l) for l in batch[0]], dtype=tf.dtypes.float32)
            code_indices = tf.ragged.constant(batch[1], dtype="int32").to_tensor()
            code_masks = tf.sequence_mask([len(l) for l in batch[1]], dtype=tf.dtypes.float32)
            buffer = buffer[len(batch[0]):]
            batch = (comment_indices, comment_masks, code_indices,
                     code_masks, tf.constant(batch[2]))
            return buffer, batch

        buffer = []
        for line in batch_data:
            if line["type"] != "BOTH":
                continue
            label = round(random.random())

            """Normal"""
            if label == 0:
                if line["type"] == "BOTH":
                    swap_dir = round(random.random())
                    comment, code = ("before_comment", "after_code") if swap_dir == 0 else (
                        "after_comment", "before_code")
                else:
                    comment, code = "before_comment", "after_code"
            else:
                if line["type"] == "BOTH":
                    swap_dir = round(random.random())
                    comment, code = ("before_comment", "before_code") if swap_dir == 0 else (
                        "after_comment", "after_code")
                else:
                    comment, code = "after_comment", "after_code"

            """swap with other data points"""
            # swap = random.choice(self.train_data)
            # line[code] = swap[code] if label == 0 else line[code]

            comment_tokens = self.vocabulary.tokenize_code(line[comment].replace("\n", "\\n"))
            code_tokens = self.vocabulary.tokenize_code("\\n".join(line[code]).replace("\n", "\\n"))

            if len(code_tokens) + len(comment_tokens) > config['data']['max_sample_size']:
                continue
            buffer.append((comment_tokens, code_tokens, label))
            if sum(sample_len(l) for l in buffer) > 50 * config['data']['max_batch_size']:
                buffer, batch = make_batch(buffer)
                yield batch

        while buffer:
            buffer, batch = make_batch(buffer)
            if not batch:
                break
            yield batch
