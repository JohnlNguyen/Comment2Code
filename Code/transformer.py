import numpy as np
import tensorflow as tf
import util

class AttentionLayer(tf.keras.layers.Layer):
	def __init__(self, attention_dim, num_heads=None, hidden_dim=None):
		super(AttentionLayer, self).__init__()
		if hidden_dim == None: hidden_dim = attention_dim
		self.attention_dim = attention_dim
		self.num_heads = 1 if num_heads is None else num_heads
		self.attn_query = tf.keras.layers.Dense(self.attention_dim, use_bias=False)
		self.attn_keys = tf.keras.layers.Dense(self.attention_dim, use_bias=False)
		self.attn_values = tf.keras.layers.Dense(self.attention_dim, use_bias=False)
		self.weight_out = tf.keras.layers.Dense(self.attention_dim if hidden_dim is None else hidden_dim, use_bias=False) # Typically, attention_dim == hidden_dim, so unless a different value is passed, we assume as much
		self.ln = [LayerNormalization(hidden_dim) for _ in range(2)]
	
	"""Applies multi-headed attention to the provided input(s).
		Supports providing just one set of states (self-attention) or separate input states to compute the keys and values over.
		Supports masked attention, in which we explicitly mask out "future" values, e.g. for generative language modeling.
	
		Note: masks are ignored if key_states are set (encoded states should be fully visible), but the reverse may not apply (e.g. in sequence tagging we self-attend without masks).
	
	Args:
		states: states to compute queries over. If key_states is None, also used to compute keys and values.
		mask: if not None, used to "mask out" values that must not be seen, e.g. padding tokens in the input or "future" tokens in a decoder/generator
		key_states: optional input states to attend over, e.g. the encoded input in neural machine translation
	"""
	def call(self, states, masks=None, key_states=None):
		# Compute key, query and value vectors, reshaped to [Batch, Heads, Time, Dim] where Dim is attention_dim//num_heads
		query, keys, values = self.compute_qkv(states, key_states)

		# Compute attention weights, and context from these
		alpha = self.get_attention_weights(query, keys, masks)
		context = tf.matmul(alpha, values)

		# Concatenate heads and transform output to hidden_dim
		context = self.concatenate_heads(context)
		context = self.weight_out(context)
		return context
	
	# Compute key, query and value vectors. If separate key_states are provided, attend over the input instead and thus assume attention is not masked
	def compute_qkv(self, states, key_states=None):
		query = self.attn_query(states) # Queries are always computed on states
		keys = self.attn_keys(states if key_states is None else key_states)
		values = self.attn_values(states if key_states is None else key_states)
		return self.reshape_for_heads(query), self.reshape_for_heads(keys), self.reshape_for_heads(values)
	
	# Split projections by heads and swap sequence and head axis to allow simpler multiplication
	def reshape_for_heads(self, value):
		value = tf.reshape(value, [value.shape[0], value.shape[1], self.num_heads, self.attention_dim//self.num_heads])
		value = tf.transpose(value, [0, 2, 1, 3])
		return value
	
	# Compute attention weights from cross-product between keys and queries (scaled, masked, softmaxed)
	def get_attention_weights(self, query, keys, masks=None):
		alpha = tf.matmul(query, keys, transpose_b=True)
		alpha *= tf.math.rsqrt(tf.cast(self.attention_dim//self.num_heads, "float32"))
		if masks is not None:
			alpha += (1.0 - masks) * tf.float32.min
		alpha = tf.nn.softmax(alpha)
		return alpha
	
	# Concatenate attention context for each head for multi-head attention
	def concatenate_heads(self, context):
		context = tf.transpose(context, [0, 2, 1, 3])
		context = tf.reshape(context, [context.shape[0], context.shape[1], self.attention_dim])
		return context

class LayerNormalization(tf.keras.layers.Layer):
	def __init__(self, hidden_dim):
		super(LayerNormalization, self).__init__()
		self.hidden_dim = hidden_dim
	
	def build(self, _):	
		self.scale = tf.Variable(tf.ones(self.hidden_dim))
		self.bias = tf.Variable(tf.zeros(self.hidden_dim))
		self.build = True
	
	def call(self, x, epsilon=1e-3):
		mean, variance = tf.nn.moments(x, -1, keepdims=True)
		norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
		return norm_x * self.scale + self.bias

class Transformer(tf.keras.layers.Layer):
	def __init__(self, embed_dim, hidden_dim, vocab_dim, attention_dim, num_layers, ff_dim, num_heads=8, dropout_rate=0.1):
		super(Transformer, self).__init__()
		random_init = tf.random_normal_initializer(stddev=hidden_dim ** -0.5)
		
		# Set up embedding and multi-headed attention layers
		self.embed = tf.Variable(random_init([vocab_dim, embed_dim]), dtype=tf.float32)
		self.self_attention = [AttentionLayer(hidden_dim, num_heads, hidden_dim) for _ in range(num_layers)]
		self.key_attention = [AttentionLayer(hidden_dim, num_heads, hidden_dim) for _ in range(num_layers)]
		
		# Layer normalization for every residual layer
		self.ln = [[LayerNormalization(hidden_dim) for _ in range(3)] for _ in range(num_layers)]
		self.ln_out = LayerNormalization(hidden_dim)
		
		# Two-layer feed-forward with wide layer in the middle
		self.ff_1 = [tf.keras.layers.Dense(ff_dim, activation="relu") for _ in range(num_layers)]
		self.ff_2 = [tf.keras.layers.Dense(hidden_dim) for _ in range(num_layers)]
		
		self.dropout_rate = dropout_rate
	
	"""Transformer language model: converts indices into hidden states through 6 layers of multi-headed attention
		To generate language from the resulting states, pass the states to "predict". Note that predict assumes input vocabulary is output vocabulary.
	
	Args:
		mask: if not None, used to mask tokens e.g. "future" tokens. See "get_sequence_mask" to get a mask specifically for this purpose
		enc_states: If not None, applies both self-attention and input attention. In that case, we never mask attention -- encoded states are assumed to be fully known
	"""
	def call(self, indices, masks=None, key_states=None, key_masks=None, training=True):
		states = tf.nn.embedding_lookup(self.embed, indices)
		states *= tf.math.sqrt(tf.cast(states.shape[-1], "float32"))
		states += util.positional_encoding(states.shape[-1], states.shape[-2])
		if training: states = tf.nn.dropout(states, rate=self.dropout_rate)
		for ix, att in enumerate(self.self_attention):
			new_states = att(self.ln[ix][0](states), masks=masks)
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			states = states + new_states
			if key_states is not None:
				new_states = self.key_attention[ix](self.ln[ix][1](states), key_states=key_states, masks=key_masks)
				if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
				states = states + new_states
			new_states = self.ff_1[ix](self.ln[ix][2](states))
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			new_states = self.ff_2[ix](new_states)
			states = states + new_states
		return self.ln_out(states)
	
	"""Returns a sequence mask in which each token can only see states up to its own position. Useful for generative language modeling (e.g. decoding)."""
	def get_sequence_mask(self, seq_len):
		return tf.sequence_mask(lengths=list(range(1, seq_len + 1)), maxlen=seq_len, dtype=tf.float32)
	
	"""Generates tokens from transformer states using the transposed embedding layer"""
	def predict(self, states):
		return util.tensor_matrix_mul(states, tf.transpose(self.embed))
