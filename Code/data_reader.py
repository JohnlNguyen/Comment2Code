﻿import numpy as np

import os
import sys
import re
import random
import json

random.seed(42)

import yaml

config = yaml.safe_load(open("config.yml"))

from vocabulary import VocabularyBuilder, BPE, SubwordTextEncoder
from pdb import set_trace
from collections import namedtuple
from util import get_data
import tensorflow as tf


class DataReader(object):
	BothBatch = namedtuple('BothBatch', 'b_tokens a_tokens code_tokens label')
	Batch = namedtuple('Batch', 'comment_tokens code_tokens label')
	AllBatch = namedtuple('AllBatch', 'b_com a_com a_cod b_cod label num_tokens')
	CodeBatch = namedtuple('CodeBatch', 'b_code a_code comment label')

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
		if vocab_config["tokenizer"] == "bpe":
			self.vocabulary = BPE(vocab_config, vocab_path=vocab_file)
		else:
			self.vocabulary = SubwordTextEncoder(vocab_config, vocab_path=vocab_file)

	def train_tokens(self):
		yield from get_data(self.train_data)

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

			# only using both changes, an ensure all
			data = [x for x in data if x['type'] == "BOTH" and all(x.values())]
			# self.get_project_idx(data)
			# self.get_file_idx(data)

			train_data = data[:int(0.95 * len(data))]
			valid_data = data[int(0.95 * len(data)):]
			test_data = []
			return train_data, valid_data, test_data

	def batcher(self, mode="training", input_type="both"):
		if input_type == "all":
			ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
				tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32),
												args=(mode, input_type,))
		elif input_type == "code":
			# b_indices, b_masks, a_indices, a_masks, c_indices, c_masks, leak_indices, leak_masks, label
			ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
				tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32),
												args=(mode, input_type,))
		elif input_type == "both":
			# b_indices, b_masks, a_indices, a_masks, c_indices, c_masks, label
			ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
				tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32), args=(mode, input_type,))
		else:
			# comment_indices, comment_masks, code_indices, code_masks, label
			ds = tf.data.Dataset.from_generator(self.batch_generator, output_types=(
				tf.int32, tf.float32, tf.int32, tf.float32, tf.float32), args=(mode, input_type,))

		ds = ds.prefetch(buffer_size=1)
		return ds

	def stats(self):
		print("Long Seq {}".format(self.long_sample_count))
		print("Num OOV {}".format(self.vocabulary.num_oov))

	def make_batch(self, buffer, input_type="both"):
		if input_type == "code":
			sample_len = lambda x: len(x.b_code) + len(x.a_code) + len(x.comment)
			buffer = self.sort_buffer(buffer, sample_len)

			batch = [[], [], [], []]
			max_seq_len = 0
			for ix, seq in enumerate(buffer):
				max_seq_len = max(max_seq_len, sample_len(seq))
				if max_seq_len * (len(batch[0]) + 1) > config['data']['max_batch_size']:
					break

				batch[0].append(seq.b_code)
				batch[1].append(seq.a_code)
				batch[2].append(seq.comment)
				batch[3].append(seq.label)

			b_code_indices, b_code_masks = self.gen_tensor(batch[0], dtype='int32')
			a_code_indices, a_code_masks = self.gen_tensor(batch[1], dtype='int32')
			comment_indices, comment_masks = self.gen_tensor(batch[2], dtype='int32')

			label = tf.constant(batch[3])

			buffer = buffer[len(batch[0]):]
			batch = (
				b_code_indices, b_code_masks, a_code_indices, a_code_masks, comment_indices,
				comment_masks, label
			)
			return buffer, batch
		elif input_type == "both":
			sample_len = lambda x: len(x.b_tokens) + len(x.a_tokens) + len(x.code_tokens)
			buffer = self.sort_buffer(buffer, sample_len)

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

			b_comment_indices, b_comment_masks = self.gen_tensor(batch[0])
			a_comment_indices, a_comment_masks = self.gen_tensor(batch[1])

			code_indices, code_masks = self.gen_tensor(batch[2])
			label = tf.constant(batch[3])

			buffer = buffer[len(batch[0]):]
			batch = (b_comment_indices, b_comment_masks, a_comment_indices, a_comment_masks, code_indices,
					 code_masks, label)
			return buffer, batch
		else:
			sample_len = lambda x: len(x.comment_tokens) + len(x.code_tokens)
			buffer = self.sort_buffer(buffer, sample_len)

			max_seq_len = 0
			batch = [[], [], []]
			for ix, seq in enumerate(buffer):
				max_seq_len = max(max_seq_len, len(seq.comment_tokens) + len(seq.code_tokens))
				if max_seq_len * (len(batch[0]) + 1) > config['data']['max_batch_size']:
					break

				batch[0].append(seq.comment_tokens)
				batch[1].append(seq.code_tokens)
				batch[2].append(seq.label)

			comment_indices, comment_masks = self.gen_tensor(batch[0])
			code_indices, code_masks = self.gen_tensor(batch[1])
			label = tf.constant(batch[2])

			buffer = buffer[len(batch[0]):]
			batch = (comment_indices, comment_masks, code_indices, code_masks, label)
			return buffer, batch

	def batch_generator(self, mode="training", input_type="both"):
		batch_data, input_type = self.setup_batch_gen(input_type, mode)

		buffer = []
		for line in batch_data:
			label = round(random.random())
			if int(line['after_line']) < 10 or int(line['before_line']) < 10: continue

			if input_type == "code":
				b_code, a_code = line['before_code'], line['after_code']
				comment = line['before_comment'] if label == 0 else line['after_comment']
				if label == 0:
					assert comment == line['before_comment']

				b_code = self.clean_code(b_code)
				a_code = self.clean_code(a_code)
				comment = self.clean_comment(comment)

				if len(comment) + min(len(b_code), len(a_code)) > config['data']['max_sample_size']:
					self.long_sample_count += 1
					continue

				b_code = self.vocabulary.transform(b_code)
				a_code = self.vocabulary.transform(a_code)
				comment = self.vocabulary.transform(comment)

				buffer.append(DataReader.CodeBatch(b_code, a_code, comment, label))

				if sum(len(l.b_code) + len(l.a_code) + len(l.comment) for l in buffer) > 75 * config['data'][
					'max_batch_size']:
					buffer, batch = self.make_batch(buffer, input_type)
					yield batch
			# using both before and after comment
			elif input_type == "both":
				item = self.gen_both_batch(line, label)
				if len(item.code_tokens) + min(len(item.b_tokens), len(item.a_tokens)) > config['data'][
					'max_sample_size']:
					self.long_sample_count += 1
					continue

				buffer.append(item)
				if sum(len(l.b_tokens) + len(l.a_tokens) + len(l.code_tokens) for l in buffer) > 75 * config['data'][
					'max_batch_size']:
					buffer, batch = self.make_batch(buffer, input_type)
					yield batch
			# swap with other a random point
			elif input_type == "random":
				if label == 0:
					comment_k, code_k = 'after_comment', 'before_code'
					swap = random.choice(self.train_data)
					comment, code = line[comment_k], swap[code_k]
				else:
					comment_k, code_k = ('after_comment', 'after_code')
					comment, code = line[comment_k], line[code_k]

				# comment_tokens, code_tokens = self.tokenize(comment, code)
				comment_tokens = self.vocabulary.transform(comment)
				code_tokens = self.vocabulary.transform(code)

				if len(code_tokens) + len(comment_tokens) > config['data']['max_sample_size']:
					self.long_sample_count += 1
					continue

				buffer.append(DataReader.Batch(comment_tokens, code_tokens, label))
				if sum(len(l.comment_tokens) + len(l.code_tokens) for l in buffer) > 50 * config['data'][
					'max_batch_size']:
					buffer, batch = self.make_batch(buffer, input_type)
					yield batch
			# swap before with after comment
			else:
				if label == 0:
					comment_k, code_k = 'after_comment', 'before_code'
					comment, code = line[comment_k], line[code_k]
				else:
					comment_k, code_k = ('after_comment', 'after_code')
					comment, code = line[comment_k], line[code_k]

				comment_tokens, code_tokens = self.tokenize(comment, code)

				if len(code_tokens) + len(comment_tokens) > config['data']['max_sample_size']:
					self.long_sample_count += 1
					continue
				buffer.append(DataReader.Batch(comment_tokens, code_tokens, label))

				if sum(len(l.comment_tokens) + len(l.code_tokens) for l in buffer) > 50 * config['data'][
					'max_batch_size']:
					buffer, batch = self.make_batch(buffer, input_type)
					yield batch

		while buffer:
			buffer, batch = self.make_batch(buffer, input_type)
			if not batch:
				break
			yield batch

	def gen_both_batch(self, line, label):
		b_comment, a_comment = line['before_comment'], line['after_comment']
		if label == 0:
			# swap_line = random.choice(self.train_data)  # TODO: need to take this out
			# code = swap_line['before_code']
			code = line['before_code']
		else:
			code = line['after_code']

		b_comment = self.clean_comment(b_comment)
		a_comment = self.clean_comment(a_comment)
		code = self.clean_code(code)

		b_comment_tokens = self.vocabulary.tokenize(b_comment)
		a_comment_tokens = self.vocabulary.tokenize(a_comment)
		code_tokens = self.vocabulary.tokenize(code)
		return DataReader.BothBatch(b_comment_tokens, a_comment_tokens, code_tokens, label)

	def get_project_idx(self, data):
		self.project_lines = {}
		for ix, row in enumerate(data):
			project_id = self.build_project_id(row)
			if project_id not in self.project_lines:
				self.project_lines[project_id] = [row]
			else:
				self.project_lines[project_id].append(row)

	def get_file_idx(self, data):
		self.file_lines = {}
		for ix, row in enumerate(data):
			project_id = row['after_path']
			if project_id not in self.file_lines:
				self.file_lines[project_id] = [row]
			else:
				self.file_lines[project_id].append(row)

	def build_project_id(self, row):
		org, project, commit, file = row['after_path'].split("#")
		project_id = "#".join([org, project])
		return project_id

	def tokenize(self, comment, code):
		comment = self.clean_comment(comment)
		code = self.clean_code(code)
		comment_tokens = self.vocabulary.tokenize(comment)
		code_tokens = self.vocabulary.tokenize(code)
		return comment_tokens, code_tokens

	def clean_code(self, code):
		return "\\n".join(code).replace("\n", "\\n")

	def clean_comment(self, comment):
		return comment.replace("\n", "\\n")

	def gen_tensor(self, data, dtype='int32'):
		return tf.ragged.constant(data, dtype=dtype).to_tensor(), tf.sequence_mask([len(l) for l in data],
																				   dtype=tf.dtypes.float32)

	def sort_buffer(self, buffer, sample_len):
		pivot = sample_len(random.choice(buffer))
		return sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))

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

	def setup_batch_gen(self, input_type, mode):
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
		return batch_data, input_type


class SimilarityDataReader(DataReader):
	def __init__(self, data_config, vocab_config, data_root, vocab_file):
		super(SimilarityDataReader, self).__init__(data_config, vocab_config, data_root, vocab_file)

	def make_batch(self, buffer, input_type="all"):
		assert str(input_type) == "all"
		sample_len = lambda x: sum([len(x.b_com), len(x.a_com), len(x.b_cod), len(x.a_cod)])
		buffer = self.sort_buffer(buffer, sample_len)

		max_seq_len = 0
		batch = [[], [], [], [], []]
		for ix, seq in enumerate(buffer):
			max_seq_len = max(max_seq_len, sample_len(seq))
			if max_seq_len * (len(batch[0]) + 1) > config['data']['max_batch_size']:
				break
			batch[0].append(seq.b_com)
			batch[1].append(seq.a_com)
			batch[2].append(seq.b_cod)
			batch[3].append(seq.a_cod)
			batch[4].append(seq.label)

		b_com_idx, b_com_masks = self.gen_tensor(batch[0], dtype='int32')
		a_com_idx, a_com_masks = self.gen_tensor(batch[1], dtype='int32')
		b_cod_idx, b_cod_masks = self.gen_tensor(batch[2], dtype='int32')
		a_cod_idx, a_cod_masks = self.gen_tensor(batch[3], dtype='int32')

		label = tf.constant(batch[4], shape=(len(batch[4]) * 4), dtype='float32')

		batch = (
			b_com_idx, b_com_masks, a_com_idx, a_com_masks, b_cod_idx, b_cod_masks, a_cod_idx, a_cod_masks, label
		)

		buffer = buffer[len(batch[0]):]
		return buffer, batch

	def batch_generator(self, mode="training", input_type="all"):
		batch_data, input_type = self.setup_batch_gen(input_type, mode)
		assert input_type == "all"

		buffer = []
		for line in batch_data:
			label = round(random.random())
			if int(line['after_line']) < 10 or int(line['before_line']) < 10:
				continue

			b_com, a_com = self.clean_comment(line['before_comment']), self.clean_comment(line['after_comment'])
			b_cod, a_cod = self.clean_code(line['before_code']), self.clean_code(line['after_code'])
			b_com, a_com = self.vocabulary.transform(b_com), self.vocabulary.transform(a_com)
			b_cod, a_cod = self.vocabulary.transform(b_cod), self.vocabulary.transform(b_cod)

			num_tokens = [len(b_com), len(a_com), len(b_cod), len(a_cod)]
			if max(num_tokens) > config['data']['max_sample_size']:
				self.long_sample_count += 1
				continue

			# swap direction
			labels = [1, 0, 0, 1]  # bb ba ab aa
			if label == 0:
				b_com, a_com = a_com, b_com
				labels = [0, 1, 1, 0]  # ab aa bb ba
				# swap_code = round(random.random())
				# if swap_code == 0:
				# 	# swap before comment with after comment
				# 	b_com, a_com = a_com, b_com
				# 	labels = [0, 1, 1, 0]  # ab aa bb ba
				# else:
				# 	b_cod, a_cod = a_cod, b_cod
				# 	labels = [0, 1, 1, 0]  # ba bb aa ab

			buffer.append(DataReader.AllBatch(b_com, a_com, b_cod, a_cod, label=labels,
											  num_tokens=sum(num_tokens)))

			if sum([x.num_tokens for x in buffer]) > 5 * config['data'][
				'max_batch_size']:
				buffer, batch = self.make_batch(buffer, input_type)
				yield batch

		while buffer:
			buffer, batch = self.make_batch(buffer, input_type)
			if not batch:
				break
			yield batch

	def random_swap(self, label, line):
		labels = [1, 0, 0, 1]  # bb ba ab aa
		if label == 0:
			swap_line = random.choice(self.train_data)
			line['before_comment'], line['after_comment'] = swap_line['before_comment'], swap_line['after_comment']

			swap_code = round(random.random())
			if swap_code == 0:
				line['after_code'] = swap_line['after_code']
				labels = [0, 0, 0, 1]
			else:
				line['before_code'] = swap_line['before_code']
				labels = [1, 0, 0, 0]
		return line, labels

	def swap_within_project(self, label, line):
		labels = [1, 0, 0, 1]  # bb ba ab aa
		if label == 0:
			proj_id = self.build_project_id(line)
			swap_line = random.choice(self.project_lines[proj_id])
			line['before_comment'], line['after_comment'] = swap_line['before_comment'], swap_line['after_comment']

			swap_code = round(random.random())
			if swap_code == 0:
				line['after_code'] = swap_line['after_code']
				labels = [0, 0, 0, 1]
			else:
				line['before_code'] = swap_line['before_code']
				labels = [1, 0, 0, 0]
		return line, labels
