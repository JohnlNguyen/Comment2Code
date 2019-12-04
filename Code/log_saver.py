import os

import tensorflow as tf


class LogSaver:

	def __init__(self, logs_path, model_name, dateset_name, mode):
		if not os.path.isdir(logs_path):
			os.makedirs(logs_path)

		self.train_writer = tf.summary.create_file_writer(
			'{}/{}/{}/{}/train/'.format(logs_path, dateset_name, model_name, mode))

		self.valid_writer = tf.summary.create_file_writer(
			'{}/{}/{}/{}/valid/'.format(logs_path, dateset_name, model_name, mode))

	def log_train(self, loss, acc, bce, global_step):
		with self.train_writer.as_default():
			tf.summary.scalar('loss', loss, step=global_step)
			tf.summary.scalar('acc', acc, step=global_step)
			tf.summary.scalar('entropy', bce, step=global_step)

	def log_valid(self, acc, bce, global_step):
		with self.valid_writer.as_default():
			tf.summary.scalar('acc', acc, step=global_step)
			tf.summary.scalar('entropy', bce, step=global_step)
