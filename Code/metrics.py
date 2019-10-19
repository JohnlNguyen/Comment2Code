import numpy as np
import tensorflow as tf

log_2_e = 1.44269504089 # Constant to convert to binary entropies

class MetricsTracker():
	def __init__(self, top_10=False):
		self.total_samples = 0
		self.flush()
	
	def flush(self, flush_totals=False):
		if flush_totals:
			self.total_samples = 0
		self.entropy = 0
		self.acc = 0.0
		self.acc_count = 0
	
	def add_observation(self, targets, predictions, loss):
		# Compute overall statistics, gathering types and predictions accordingly
		num_samples = targets.shape[0]
		self.entropy += log_2_e * loss.numpy() * num_samples
		self.acc += tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(targets, predictions)).numpy()
		self.acc_count += num_samples
		self.total_samples += int(num_samples)
	
	def get_stats(self):
		loss = self.entropy / self.acc_count if self.acc_count > 0 else 0
		acc = self.acc / self.acc_count if self.acc_count > 0 else 0
		return self.total_samples, "{0:.3f}".format(loss), "{0:.2%}".format(acc)
