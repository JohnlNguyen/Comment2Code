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
from collections import namedtuple
import tensorflow as tf


class DataReader(object):
    BothBatch = namedtuple('BothBatch', 'b_tokens a_tokens code_tokens label')
    Batch = namedtuple('Batch', 'comment_tokens code_tokens label')

    def __init__(self, data_config, vocab_config, data_root, vocab_file):
        self.config = data_config
        self.train_data, self.valid_data, self.test_data = self.read(data_root)
        print("%d lines" % len(self.train_data))
        self.get_vocab(vocab_config, vocab_file)

        # Limit held-out data size
        if sum(len(l) for l in self.valid_data) > 1000000:
            random.shuffle(self.valid_data)
            self.valid_data = self.valid_data[:250]

        self.sample_len = lambda l: len(l[0]) + len(l[1])

        # stats for data
        self.long_sample_count = 0

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
            with open(data_root, encoding='utf-8', errors='ignore') as f:
                data = json.load(f)

            # subset data
            percent = float(self.config['percent'])
            data = data[:int(len(data) * percent)]

            # only using both changes
            data = [x for x in data if x['type'] == "BOTH"]

            train_data = data[:int(0.95 * len(data))]
            valid_data = data[int(0.95 * len(data)): int(len(data))]
            test_data = []
            return train_data, valid_data, test_data

    def batcher(self, mode="training", input_type="both"):
        if input_type == "both":
            # b_indices, b_masks, a_indices, a_masks, c_indices, c_masks, label
            ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
                tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32), args=(mode, input_type,))
        else:
            # comment_indices, comment_masks, code_indices, code_masks, label
            ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
                tf.int32, tf.float32, tf.int32, tf.float32, tf.float32), args=(mode, input_type,))

        ds = ds.prefetch(1)
        return ds

    def stats(self):
        print("Long Seq {}".format(self.long_sample_count))
        print("Num OOV {}".format(self.vocabulary.num_oov))

    def make_batch(self, buffer, input_type="both"):

        if input_type == "both":
            sample_len = lambda x: len(x.b_tokens) + len(x.a_tokens) + len(x.code_tokens)

            pivot = sample_len(random.choice(buffer))
            buffer = sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))

            batch = [[], [], [], []]
            max_seq_len = 0
            for ix, seq in enumerate(buffer):
                max_seq_len = max(max_seq_len, sample_len(seq))
                if max_seq_len * (len(batch[0]) + 1) > config['data']['max_batch_size']:
                    break

                batch[0].append([self.vocabulary.vocab_key(s) for s in seq.b_tokens])
                batch[1].append([self.vocabulary.vocab_key(s) for s in seq.a_tokens])
                batch[2].append([self.vocabulary.vocab_key(s) for s in seq.code_tokens])
                batch[3].append(seq.label)

            b_comment_indices = tf.ragged.constant(batch[0], dtype="int32").to_tensor()
            b_comment_masks = tf.sequence_mask([len(l) for l in batch[0]], dtype=tf.dtypes.float32)

            a_comment_indices = tf.ragged.constant(batch[1], dtype="int32").to_tensor()
            a_comment_masks = tf.sequence_mask([len(l) for l in batch[1]], dtype=tf.dtypes.float32)

            code_indices = tf.ragged.constant(batch[2], dtype="int32").to_tensor()
            code_masks = tf.sequence_mask([len(l) for l in batch[2]], dtype=tf.dtypes.float32)

            buffer = buffer[len(batch[0]):]
            batch = (b_comment_indices, b_comment_masks, a_comment_indices, a_comment_masks, code_indices,
                     code_masks, tf.constant(batch[3]))
            return buffer, batch
        else:
            sample_len = lambda x: len(x.comment_tokens) + len(x.code_tokens)
            max_seq_len = 0
            batch = [[], [], []]
            for ix, seq in enumerate(buffer):
                max_seq_len = max(max_seq_len, sample_len(seq))
                if max_seq_len * (len(batch[0]) + 1) > config['data']['max_batch_size']:
                    break

                batch[0].append([self.vocabulary.vocab_key(s) for s in seq.comment_tokens])
                batch[1].append([self.vocabulary.vocab_key(s) for s in seq.code_tokens])
                batch[2].append(seq.label)

            comment_indices = tf.ragged.constant(batch[0], dtype="int32").to_tensor()
            comment_masks = tf.sequence_mask([len(l) for l in batch[0]], dtype=tf.dtypes.float32)

            code_indices = tf.ragged.constant(batch[1], dtype="int32").to_tensor()
            code_masks = tf.sequence_mask([len(l) for l in batch[1]], dtype=tf.dtypes.float32)
            buffer = buffer[len(batch[0]):]
            batch = (comment_indices, comment_masks, code_indices,
                     code_masks, tf.constant(batch[2]))
            return buffer, batch

    def batch_generator(self, mode="training", input_type="both"):
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if isinstance(input_type, bytes):
            input_type = input_type.decode("utf-8")

        if mode == "training":
            batch_data = self.train_data
            random.shuffle(batch_data)
        elif mode == "valid":
            batch_data = self.valid_data
        else:
            batch_data = self.test_data

        buffer = []
        for line in batch_data:
            label = round(random.random())

			# using both before and after comment
            if input_type == "both":
                item = self.gen_both_batch(line, label)

                if len(item.code_tokens) + min(len(item.b_tokens), len(item.a_tokens)) > config['data'][
                        'max_sample_size']:
                    self.long_sample_count += 1
                    continue

                buffer.append(item)
                if sum(len(l.b_tokens) + len(l.a_tokens) + len(l.code_tokens) for l in buffer) > 50 * config['data']['max_batch_size']:
                    buffer, batch = self.make_batch(buffer, input_type)
                    yield batch
			# swap with other a random point
            elif input_type == "random":
                comment, code = DataReader.swap_keys(label, line)
                swap = random.choice(self.train_data)
                line[code] = swap[code] if label == 0 else line[code]
                comment_tokens, code_tokens = self.tokenize(comment, code)

                if len(code_tokens) + len(comment_tokens) > config['data']['max_sample_size']:
                    self.long_sample_count += 1
                    continue

                buffer.append(DataReader.Batch(comment_tokens, code_tokens, label))
                if sum(len(l.comment_tokens) + len(l.code_tokens) for l in buffer) > 50 * config['data']['max_batch_size']:
                    buffer, batch = self.make_batch(buffer, input_type)
                    yield batch
			# swap before with after comment
            else:
                comment_k, code_k = DataReader.swap_keys(label, line)
                comment, code = line[comment_k], line[code_k]
                comment_tokens, code_tokens = self.tokenize(comment, code)

                if len(code_tokens) + len(comment_tokens) > config['data']['max_sample_size']:
                    self.long_sample_count += 1
                    continue
                buffer.append(DataReader.Batch(comment_tokens, code_tokens, label))

                if sum(len(l.comment_tokens) + len(l.code_tokens) for l in buffer) > 50 * config['data']['max_batch_size']:
                    buffer, batch = self.make_batch(buffer, input_type)
                    yield batch

        while buffer:
            buffer, batch = self.make_batch(buffer, input_type)
            if not batch:
                break
            yield batch

    def gen_both_batch(self, line, label):

        b_comment, a_comment = line['before_comment'], line['after_comment']
        line = random.choice(self.train_data)
        code = line['before_code'] if label == 0 else line['after_code']

        b_comment_tokens = self.vocabulary.tokenize(b_comment.replace("\n", "\\n"))
        a_comment_tokens = self.vocabulary.tokenize(b_comment.replace("\n", "\\n"))
        code_tokens = self.vocabulary.tokenize("\\n".join(code).replace("\n", "\\n"))

        return DataReader.BothBatch(b_comment_tokens, a_comment_tokens, code_tokens, label)

    def tokenize(self, comment, code):
        comment_tokens = self.vocabulary.tokenize(comment.replace("\n", "\\n"))
        code_tokens = self.vocabulary.tokenize("\\n".join(code).replace("\n", "\\n"))
        return comment_tokens, code_tokens

    @classmethod
    def swap_keys(cls, label, line):
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
        return comment, code
