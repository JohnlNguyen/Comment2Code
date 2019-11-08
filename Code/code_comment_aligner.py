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
import tensorflow as tf

from pdb import set_trace
from transformer import Transformer, AttentionLayer
from metrics import MetricsTracker
from data_reader import DataReader
import util

random.seed(41)
config = yaml.safe_load(open("config.yml"))
pp = pprint.PrettyPrinter(indent=2)
best_eval_acc = np.NINF


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
	# Extract arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("data", help="Path to training data")
	ap.add_argument("-v", "--vocabulary", required=False, help="Path to vocabulary files")
	args = ap.parse_args()
	print("Using configuration: ")
	log_and_print(pprint.pformat(config))

	data = DataReader(config["data"], config["vocabulary"], args.data, vocab_file=args.vocabulary)
	if config["training"]["model"] == "transformer":
		model = TransformerModel(config["transformer"], data.vocabulary.vocab_dim)
	elif config["training"]["model"] == "baseline" or config["training"]["model"] == "rnn":
		model = BaselineModel(config["baseline"], data.vocabulary.vocab_dim)
	elif config["training"]["model"] == "multimodal":
		model = MultiModalTransformerModel(config["multimodal"], data.vocabulary.vocab_dim)
	elif config["training"]["model"] == "shared_transformer":
		model = SharedTransformerModel(config["transformer"], data.vocabulary.vocab_dim)
	elif config["training"]["model"] == "similarity_transformer":
		model = SimilarityTransformerModel(config["transformer"], data.vocabulary.vocab_dim)
	elif config["training"]["model"] == "code_transformer":
		model = CodeTransformerModel(config["transformer"], data.vocabulary.vocab_dim)
	train(model, data)


def train(model, data):
	# Declare the learning rate as a function to include it in the saved state
	def get_learning_rate():
		if "transformer" not in config["training"]["model"]:
			return tf.constant(config["training"]["lr"])
		return tf.constant(config["training"]["lr"])

	optimizer = tf.optimizers.Adam(get_learning_rate)

	total_batches = 0
	is_first = True
	for epoch in range(config["training"]["num_epochs"]):
		log_and_print("Epoch: {}".format(epoch + 1))
		metrics = MetricsTracker()
		mbs = 0
		for batch in data.batcher(mode="training", input_type=config["training"]["input_type"]):
			mbs += 1
			total_batches += 1

			# Run through one batch to init variables
			if is_first:
				model(*batch[:-1])
				is_first = False
				log_and_print("Model initialized, training {:,} parameters".format(
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
				log_and_print("MB: {0}, lr: {1:.1e}: samples: {2:,}, entropy: {3}, acc: {4} loss: {5:.4f}".format(
					mbs, lr, *metrics.get_stats(), loss))
				metrics.flush()

		# Run a validation pass at the end of every epoch
		log_and_print("Validation: samples: {0}, entropy: {1}, accs: {2}".format(
			*eval(model, data)))
	log_and_print(
		"Test: samples: {0}, entropy: {1}, accs: {2}".format(*eval(model, data, validate=False)))


def eval(model, data, validate=True):
	mbs = 0
	metrics = MetricsTracker()
	valid_preds, valid_target = np.array([]), np.array([])
	global best_eval_acc
	for batch in data.batcher(mode="valid" if validate else "test", input_type=config["training"]["input_type"]):
		mbs += 1
		preds = model(*batch[:-1])
		metrics.add_observation(batch[-1], preds, get_loss(batch[-1], preds))
		valid_preds = np.append(valid_preds, preds.numpy())
		valid_target = np.append(valid_target, batch[-1].numpy())

	acc = metrics.get_acc()
	if config["training"]["save_valid"] and best_eval_acc < acc:
		with open("valid_{}.pkl".format(config["training"]["model"]), 'wb') as f:
			pickle.dump([list(valid_preds), list(valid_target)], f)
		best_eval_acc = acc
	return metrics.get_stats()


def log_loss(targets, predictions):
	return tf.reduce_mean(
		-tf.math.log(1e-6 + predictions) * targets + -tf.math.log(1e-6 + 1 - predictions) * (1 - targets))


def get_cp_loss(targets, predictions):
	"""
	Confidence Penalty Loss
	https://arxiv.org/pdf/1701.06548.pdf
	"""
	log_loss = tf.losses.binary_crossentropy(targets, predictions)
	entropy = -tf.reduce_mean(predictions * tf.math.log(1e-6 + predictions))
	cp = float(config["training"]["beta"]) * entropy
	return log_loss + cp


def get_loss(targets, predictions):
	if config["training"]["loss_function"] == "cp":
		return get_cp_loss(targets, predictions)
	else:
		return log_loss(targets, predictions)


class MultiModalTransformerModel(tf.keras.layers.Layer):

	def __init__(self, model_config, token_vocab_dim):
		super(MultiModalTransformerModel, self).__init__()
		self.model_config = model_config
		self.token_vocab_dim = token_vocab_dim

		self.code_transformer = self.build_transformer()
		self.b_comment_transformer = self.build_transformer()
		self.a_comment_transformer = self.build_transformer()

		# Projection layers
		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
		self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

	def call(self, b_comment_indices, b_comment_masks, a_comment_indices, a_comment_masks, code_indices, code_masks):
		# Set up masks for our three transformers
		code_self_masks = tf.reshape(code_masks, [code_masks.shape[0], 1, 1, code_masks.shape[1]])

		b_code_key_masks = code_self_masks * tf.reshape(b_comment_masks,
														[b_comment_masks.shape[0], 1, b_comment_masks.shape[1], 1])
		a_code_key_masks = code_self_masks * tf.reshape(a_comment_masks,
														[a_comment_masks.shape[0], 1, a_comment_masks.shape[1], 1])

		b_comment_self_masks = tf.reshape(
			b_comment_masks, [b_comment_masks.shape[0], 1, 1, b_comment_masks.shape[1]])
		a_comment_self_masks = tf.reshape(
			a_comment_masks, [a_comment_masks.shape[0], 1, 1, a_comment_masks.shape[1]])

		# Compute code self-attention states
		code_states = self.code_transformer(code_indices, masks=code_self_masks)

		# Compute before comment self+code-attention states
		b_states = self.b_comment_transformer(
			b_comment_indices, masks=b_comment_self_masks, key_states=code_states, key_masks=b_code_key_masks)
		# Max pool
		b_states = tf.reduce_max(b_states, axis=1)

		a_states = self.a_comment_transformer(
			a_comment_indices, masks=a_comment_self_masks, key_states=code_states, key_masks=a_code_key_masks)
		a_states = tf.reduce_max(a_states, axis=1)

		# concat before and after
		states = tf.concat([b_states, a_states], axis=-1)

		# prediction
		states = self.fc1(states)
		preds = self.classify(states)
		preds = tf.squeeze(preds, -1)
		return preds

	def build_transformer(self):
		return Transformer(self.model_config["embed_dim"], self.model_config["hidden_dim"], self.token_vocab_dim,
						   self.model_config["attention_dim"], self.model_config["num_layers"],
						   self.model_config["ff_dim"],
						   self.model_config["num_heads"], self.model_config["dropout_rate"])


class TransformerModel(tf.keras.layers.Layer):

	def __init__(self, model_config, token_vocab_dim):
		super(TransformerModel, self).__init__()
		self.code_transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim,
											model_config[
												"attention_dim"], model_config["num_layers"], model_config["ff_dim"],
											model_config["num_heads"], model_config["dropout_rate"])
		self.comment_transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim,
											   model_config[
												   "attention_dim"], model_config["num_layers"],
											   model_config["ff_dim"],
											   model_config["num_heads"], model_config["dropout_rate"])

		self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

	def call(self, comment_indices, comment_masks, code_indices, code_masks):
		# Set up masks for our two transformers
		comment_self_masks = tf.reshape(
			comment_masks, [comment_masks.shape[0], 1, 1, comment_masks.shape[1]])
		code_self_masks = tf.reshape(code_masks, [code_masks.shape[0], 1, 1, code_masks.shape[1]])
		code_key_masks = code_self_masks * \
						 tf.reshape(comment_masks, [comment_masks.shape[0], 1, comment_masks.shape[1], 1])

		# Compute code self-attention states
		code_states = self.code_transformer(code_indices, masks=code_self_masks)

		# Compute comment self+code-attention states
		states = self.comment_transformer(
			comment_indices, masks=comment_self_masks, key_states=code_states, key_masks=code_key_masks)

		# Max-pool states and project to classify
		states = tf.reduce_max(states, 1)
		preds = self.classify(states)
		preds = tf.squeeze(preds, -1)
		return preds


class SimilarityTransformerModel(TransformerModel):
	def __init__(self, model_config, token_vocab_dim):
		super(SimilarityTransformerModel, self).__init__(model_config, token_vocab_dim)

		self.self_attention = [
			AttentionLayer(attention_dim=model_config["attention_dim"], num_heads=model_config["num_heads"],
						   hidden_dim=model_config["hidden_dim"])
			for _ in range(3)]

		self.fc_1 = tf.keras.layers.Dense(512, activation='relu')

	def call(self, b_comment_indices, b_comment_masks, a_comment_indices, a_comment_masks, b_code_indices, b_code_masks,
			 a_code_indices, a_code_masks):
		b_code_self_masks = tf.reshape(b_code_masks, [b_code_masks.shape[0], 1, 1, b_code_masks.shape[1]])
		a_code_self_masks = tf.reshape(a_code_masks, [a_code_masks.shape[0], 1, 1, a_code_masks.shape[1]])

		b_b_code_key_masks = b_code_self_masks * tf.reshape(b_comment_masks,
															[b_comment_masks.shape[0], 1, b_comment_masks.shape[1], 1])
		b_a_code_key_masks = a_code_self_masks * tf.reshape(b_comment_masks,
															[b_comment_masks.shape[0], 1, b_comment_masks.shape[1], 1])

		a_b_code_key_masks = b_code_self_masks * tf.reshape(a_comment_masks,
															[a_comment_masks.shape[0], 1, a_comment_masks.shape[1], 1])
		a_a_code_key_masks = a_code_self_masks * tf.reshape(a_comment_masks,
															[a_comment_masks.shape[0], 1, a_comment_masks.shape[1], 1])

		b_comment_self_masks = tf.reshape(
			b_comment_masks, [b_comment_masks.shape[0], 1, 1, b_comment_masks.shape[1]])
		a_comment_self_masks = tf.reshape(
			a_comment_masks, [a_comment_masks.shape[0], 1, 1, a_comment_masks.shape[1]])

		b_code_states = self.code_transformer(b_code_indices, masks=b_code_self_masks)  # [batch, seq len, hidden dim]
		a_code_states = self.code_transformer(a_code_indices, masks=a_code_self_masks)  # [batch, seq len, hidden dim]

		b_b_states = self.comment_transformer(
			b_comment_indices, masks=b_comment_self_masks, key_states=b_code_states,
			key_masks=b_b_code_key_masks)  # [batch, b code len, hidden dim]
		b_b_states = tf.reduce_max(b_b_states, axis=1)  # [batch, hidden dim]

		b_a_states = self.comment_transformer(
			b_comment_indices, masks=b_comment_self_masks, key_states=a_code_states,
			key_masks=b_a_code_key_masks)  # [batch, seq len, hidden dim]
		b_a_states = tf.reduce_max(b_a_states, axis=1)  # [batch, hidden dim]

		a_b_states = self.comment_transformer(
			a_comment_indices, masks=a_comment_self_masks, key_states=b_code_states,
			key_masks=a_b_code_key_masks)  # [batch, seq len, hidden dim]
		a_b_states = tf.reduce_max(a_b_states, axis=1)  # [batch, hidden dim]

		a_a_states = self.comment_transformer(
			a_comment_indices, masks=a_comment_self_masks, key_states=a_code_states, key_masks=a_a_code_key_masks)
		a_a_states = tf.reduce_max(a_a_states, axis=1)

		b_b_states = tf.expand_dims(b_b_states, axis=1)
		b_a_states = tf.expand_dims(b_a_states, axis=1)
		a_b_states = tf.expand_dims(a_b_states, axis=1)
		a_a_states = tf.expand_dims(a_a_states, axis=1)
		states = tf.concat([b_b_states, b_a_states, a_b_states, a_a_states], axis=1)  # [B, 4, hidden dim]

		for attention in self.self_attention:
			states = attention(states)

		states = self.fc_1(states)  # [B, 4, 512]
		preds = self.classify(states)  # [B, 4, 1]
		preds = tf.reshape(preds, shape=[-1]) # flatten into a B x 4 vector
		return preds


class SharedTransformerModel(TransformerModel):

	def __init__(self, model_config, token_vocab_dim):
		super(SharedTransformerModel, self).__init__(model_config, token_vocab_dim)

		self.self_attention = [
			AttentionLayer(attention_dim=model_config["attention_dim"], num_heads=model_config["num_heads"],
						   hidden_dim=model_config["hidden_dim"])
			for _ in range(3)]

		self.fc_1 = tf.keras.layers.Dense(512, activation='relu')

	def call(self, b_comment_indices, b_comment_masks, a_comment_indices, a_comment_masks, code_indices, code_masks):
		# Set up masks for our three transformers
		code_self_masks = tf.reshape(code_masks, [code_masks.shape[0], 1, 1, code_masks.shape[1]])

		b_code_key_masks = code_self_masks * tf.reshape(b_comment_masks,
														[b_comment_masks.shape[0], 1, b_comment_masks.shape[1], 1])
		a_code_key_masks = code_self_masks * tf.reshape(a_comment_masks,
														[a_comment_masks.shape[0], 1, a_comment_masks.shape[1], 1])

		b_comment_self_masks = tf.reshape(
			b_comment_masks, [b_comment_masks.shape[0], 1, 1, b_comment_masks.shape[1]])
		a_comment_self_masks = tf.reshape(
			a_comment_masks, [a_comment_masks.shape[0], 1, 1, a_comment_masks.shape[1]])

		# Compute code self-attention states
		code_states = self.code_transformer(code_indices, masks=code_self_masks)

		# Compute before comment self+code-attention states
		b_states = self.comment_transformer(
			b_comment_indices, masks=b_comment_self_masks, key_states=code_states, key_masks=b_code_key_masks)
		b_states = tf.reduce_max(b_states, axis=1)

		a_states = self.comment_transformer(
			a_comment_indices, masks=a_comment_self_masks, key_states=code_states, key_masks=a_code_key_masks)
		a_states = tf.reduce_max(a_states, axis=1)

		states = tf.concat([b_states, a_states], axis=-1)

		# prediction
		states = self.fc_1(states)
		preds = self.classify(states)
		preds = tf.squeeze(preds, -1)
		return preds


class CodeTransformerModel(SharedTransformerModel):
	def __init__(self, model_config, token_vocab_dim):
		super(CodeTransformerModel, self).__init__(model_config, token_vocab_dim)

	def call(self, b_code_indices, b_code_masks, a_code_indices, a_code_masks, comment_indices, comment_masks):
		# setting up masks
		b_code_self_masks = tf.reshape(b_code_masks, [b_code_masks.shape[0], 1, 1, b_code_masks.shape[1]])
		a_code_self_masks = tf.reshape(a_code_masks, [a_code_masks.shape[0], 1, 1, a_code_masks.shape[1]])

		comment_self_masks = tf.reshape(
			comment_masks, [comment_masks.shape[0], 1, 1, comment_masks.shape[1]])

		a_comment_key_masks = comment_self_masks * tf.reshape(a_code_masks,
															  [a_code_masks.shape[0], 1, a_code_masks.shape[1], 1])

		b_comment_key_masks = comment_self_masks * tf.reshape(b_code_masks,
															  [b_code_masks.shape[0], 1, b_code_masks.shape[1], 1])

		# comment
		comment_states = self.comment_transformer(comment_indices, masks=comment_self_masks)

		# before code and maxpool
		b_states = self.code_transformer(b_code_indices, masks=b_code_self_masks, key_states=comment_states,
										 key_masks=b_comment_key_masks)
		b_states = tf.reduce_max(b_states, axis=1)

		# after code and maxpool
		a_states = self.code_transformer(a_code_indices, masks=a_code_self_masks, key_states=comment_states,
										 key_masks=a_comment_key_masks)
		a_states = tf.reduce_max(a_states, axis=1)

		# concat
		states = tf.concat([b_states, a_states], axis=-1)

		# prediction
		states = self.fc1(states)
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
		preds = self.classify(states)
		preds = tf.squeeze(preds, -1)
		return preds


def log_and_print(msg):
	print(msg)
	logfile = os.path.join("../data", config["training"]["model"] + ".log")
	with open(logfile, 'a+') as f:
		f.write(msg + '\n')

if __name__ == '__main__':
	main()
