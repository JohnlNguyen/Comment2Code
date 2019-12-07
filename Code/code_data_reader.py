import tensorflow as tf
import tensorflow_datasets as tfds
from collections import namedtuple
from pdb import set_trace
import yaml
import numpy as np

import os
import sys
import re
import random
import json

random.seed(42)


config = yaml.safe_load(open("config.yml"))


class CodeDataReader(object):
    CodeBatch = namedtuple('CodeBatch', 'b_code a_code comment label')

    def __init__(self, data_config, data_root, test_file=None, vocab_file=None):
        self.config = data_config
        self.train_data, self.valid_data, self.test_data = self.read(
            data_root, test_file)
        self.filter_max_length()

        print("%d lines" % len(self.train_data))
        self.get_vocab(vocab_file)

        # Limit held-out data size
        if sum(len(l) for l in self.valid_data) > 1000000:
            random.shuffle(self.valid_data)
            self.valid_data = self.valid_data[:250]

        self.sample_len = lambda l: len(l[0]) + len(l[1])

    def filter_max_length(self):
        def is_long(x): return len(x['before_comment']) + len(x['before_code']) + len(
            x['after_code']) > config['data']['max_sample_size']

        self.train_data = [row for row in self.train_data if not is_long(row)]
        self.valid_data = [row for row in self.valid_data if not is_long(row)]
        # self.test_data = [row for row in self.test_data if not is_long(row)]

    def get_vocab(self, vocab_path):
        if not vocab_path:
            self.vocabulary = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                self.generator(), target_vocab_size=2 ** 13)
            self.vocabulary.save_to_file('./data/vocab_code_java')
        else:
            self.vocabulary = tfds.features.text.SubwordTextEncoder.load_from_file(
                vocab_path)

    def generator(self):
        for line in self.train_data + self.valid_data:
            yield line['before_comment']
            yield line['after_comment']
            yield self.clean_code(line['before_code'])
            yield self.clean_code(line['after_code'])

    def read(self, data_root, test_file=None):
        with open(data_root, encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        data = [row for row in data if row['type']
                == "BOTH" or row['type'] == "CODE"]

        test_data = []
        if test_file is not None:
            with open(test_file, encoding='utf-8', errors='ignore') as f:
                test_data = json.load(f)

        # subset data
        percent = float(self.config['percent'])
        data = data[:int(len(data) * percent)]
        train_data = data[:int(0.90 * len(data))]
        valid_data = data[int(0.90 * len(data)):]
        return train_data, valid_data, test_data

    def batcher(self, mode="training"):
        # b_indices, b_masks, a_indices, a_masks, c_indices, c_masks, label
        ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
            tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32),
            args=(mode,))
        ds = ds.prefetch(buffer_size=1)
        return ds

    def make_batch(self, buffer):
        def sample_len(x): return len(x.b_code) + \
            len(x.a_code) + len(x.comment)
        buffer = self.sort_buffer(buffer, sample_len)

        batch = [[], [], [], []]
        max_seq_len = 0
        for ix, seq in enumerate(buffer):
            max_seq_len = max(max_seq_len, sample_len(seq))

            if (len(batch[0]) > 0 and len(batch[1]) > 0 and len(
                    batch[2]) > 0 and len(batch[3]) > 0) and max_seq_len * (len(batch[0]) + 1) > config['data']['max_batch_size']:
                break

            batch[0].append(seq.b_code)
            batch[1].append(seq.a_code)
            batch[2].append(seq.comment)
            batch[3].append(seq.label)
        assert len(batch[0]) > 0 and len(batch[1]) > 0 and len(
            batch[2]) > 0 and len(batch[3]) > 0

        b_code_indices, b_code_masks = self.gen_tensor(batch[0], dtype='int32')
        a_code_indices, a_code_masks = self.gen_tensor(batch[1], dtype='int32')
        comment_indices, comment_masks = self.gen_tensor(
            batch[2], dtype='int32')

        label = tf.constant(batch[3])

        buffer = buffer[len(batch[0]):]
        batch = (
            b_code_indices, b_code_masks, a_code_indices, a_code_masks, comment_indices,
            comment_masks, label
        )
        return buffer, batch

    def batch_generator(self, mode="training"):
        batch_data = self.setup_batch_gen(mode)
        buffer = []

        for line in batch_data:
            assert line['type'] == "BOTH" or line['type'] == "CODE"
            label = 1 if line['type'] == 'BOTH' else 0

            b_code, a_code = line['before_code'], line['after_code']
            comment = line['before_comment']

            b_code = self.clean_code(b_code)
            a_code = self.clean_code(a_code)
            if len(comment) + len(b_code) + len(a_code) > config['data']['max_sample_size']:
                continue

            b_code = self.vocabulary.encode(b_code)
            a_code = self.vocabulary.encode(a_code)
            comment = self.vocabulary.encode(comment)

            buffer.append(CodeDataReader.CodeBatch(
                b_code, a_code, comment, label))

            if len(buffer) > 0 and sum(len(l.b_code) + len(l.a_code) + len(l.comment) for l in buffer) > config['data'][
                    'max_batch_size']:
                buffer, batch = self.make_batch(buffer)
                yield batch

        while buffer:
            buffer, batch = self.make_batch(buffer)
            if not batch:
                break
            yield batch

    def clean_code(self, code):
        return "\\n".join(code).replace("\n", "\\n")

    def gen_tensor(self, data, dtype='int32'):
        return tf.ragged.constant(data, dtype=dtype).to_tensor(), tf.sequence_mask([len(l) for l in data],
                                                                                   dtype=tf.dtypes.float32)

    def sort_buffer(self, buffer, sample_len):
        pivot = sample_len(random.choice(buffer))
        return sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))

    def setup_batch_gen(self, mode):
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")

        if mode == "training":
            # batch_data = self.train_data
            both = [row for row in self.train_data if row["type"] == "BOTH"]
            code = [row for row in self.train_data if row["type"] == "CODE"]
            batch_data = both + random.sample(code, len(both))
            random.shuffle(batch_data)
        elif mode == "valid":
            both = [row for row in self.valid_data if row["type"] == "BOTH"]
            code = [row for row in self.valid_data if row["type"] == "CODE"]
            batch_data = both + random.sample(code, len(both))
        elif mode == "test":
            batch_data = [{**row, **{
                'type': "BOTH" if row['label'] == "1" else "CODE"}
            }
                for row in self.test_data
            ]
        return batch_data
