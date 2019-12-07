import re
import string

import numpy as np
import tensorflow as tf

# Based on https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py
pos_cache = None


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    global pos_cache
    if pos_cache != None and pos_cache[1] == dim:
        if pos_cache[0] == sentence_length:
            return pos_cache[2]
        elif pos_cache[0] > sentence_length:
            return pos_cache[2][:sentence_length]
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim)
                            for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    pos_enc = tf.constant(encoded_vec.reshape(
        [sentence_length, dim]), dtype=dtype)
    pos_cache = (sentence_length, dim, pos_enc)
    return pos_enc


def compute_transformer_learning_rate(base_lr, hidden_dim, warmup_steps, curr_step):
    learning_rate = base_lr * (hidden_dim ** -0.5)
    learning_rate *= tf.minimum(1.0, curr_step / warmup_steps)
    learning_rate *= tf.math.rsqrt(tf.cast(tf.maximum(curr_step,
                                                      warmup_steps), "float32"))
    return learning_rate


def tensor_matrix_mul(t, m):
    return tf.reshape(tf.reshape(t, [-1, t.shape[-1]]) @ m, [-1, t.shape[1], m.shape[-1]])


def merge(map, key, value, wrap_fn=None):
    if key in map:
        if isinstance(map[key], int) or isinstance(map[key], float):
            map[key] += value
        elif isinstance(map[key], list):
            map[key].append(value)
        elif isinstance(map[key], set):
            map[key].add(value)
        else:
            print("Unsure how to add", value, "to", map[key])
    else:
        map[key] = value if wrap_fn is None else wrap_fn(value)


def get_data(data):
    for line in data:
        yield line['before_comment']
        yield line['after_comment']
        yield line['before_code']
        yield line['after_code']
