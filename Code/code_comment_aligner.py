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
from data_reader import DataReader, SimilarityDataReader
from log_saver import LogSaver
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
    ap.add_argument("-v", "--vocabulary", required=False,
                    help="Path to vocabulary files")
    args = ap.parse_args()
    print("Using configuration: ")
    log_and_print(pprint.pformat(config))
    if config["training"]["input_type"] == "all":
        data = SimilarityDataReader(
            config["data"], config["vocabulary"], args.data, vocab_file=args.vocabulary)
    else:
        data = DataReader(config["data"], config["vocabulary"],
                          args.data, vocab_file=args.vocabulary)

    if config["training"]["model"] == "transformer":
        model = TransformerModel(
            config["transformer"], data.vocabulary.vocab_dim)
    elif config["training"]["model"] == "baseline" or config["training"]["model"] == "rnn":
        model = BaselineModel(config["baseline"], data.vocabulary.vocab_dim)
    elif config["training"]["model"] == "shared_transformer":
        model = SharedTransformerModel(
            config["transformer"], data.vocabulary.vocab_dim)
    elif config["training"]["model"] == "similarity_transformer":
        model = SimilarityTransformerModel(
            config["transformer"], data.vocabulary.vocab_dim)
    elif config["training"]["model"] == "code_transformer":
        model = CodeTransformerModel(
            config["transformer"], data.vocabulary.vocab_dim)
    train(model, data)


def create_checkpoint(config, model, optimizer):
    checkpoint_path = config["training"]["checkpoint"]
    ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)
    return ckpt, ckpt_manager


def train(model, data):
    # Declare the learning rate as a function to include it in the saved state
    def get_learning_rate():
        if "transformer" not in config["training"]["model"]:
            return tf.constant(config["training"]["lr"])
        return tf.constant(config["training"]["lr"])

    optimizer = tf.optimizers.Adam(get_learning_rate)
    log = LogSaver(config["training"]["log_dir"], config["training"]["model"], config["training"]["input_type"],
                   "direction")

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
        for batch in data.batcher(mode="training", input_type=config["training"]["input_type"]):
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
        loss = get_loss(batch[-1], preds)
    # Collect gradients, clip and apply
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 0.25)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, preds


def eval(model, data, validate=True):
    mbs = 0
    metrics = MetricsTracker()
    valid_preds, valid_target = np.array([]), np.array([])
    global best_eval_acc
    for batch in data.batcher(mode="valid" if validate else "test", input_type=config["training"]["input_type"]):
        mbs += 1
        preds = model(*batch[:-1], training=False)
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
    if config["training"]["loss_function"] == "constrastive":
        return contrastive_loss(targets, predictions)
    if config["training"]["loss_function"] == "mse":
        return tf.keras.losses.mse(targets, predictions)
    if config["training"]["loss_function"] == "hinge":
        return tf.keras.losses.hinge(targets, predictions)
    if config["training"]["loss_function"] == "cp":
        return get_cp_loss(targets, predictions)
    else:
        return tf.losses.binary_crossentropy(targets, predictions, label_smoothing=.01)


def euclidean_distance(x1, x2):
    return tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=1, keepdims=True))


def manhattan_similarity(x1, x2):
    manhattan_sim = tf.exp(-tf.reduce_sum(tf.abs(x1 - x2),
                                          axis=1, keepdims=True))
    return manhattan_sim


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    term1 = y_true * tf.square(y_pred)
    term2 = (1 - y_true) * tf.square(tf.maximum((margin - y_pred), 0))
    return tf.reduce_sum(tf.add(term1, term2) / 2)


class TransformerModel(tf.keras.Model):

    def __init__(self, model_config, token_vocab_dim):
        super(TransformerModel, self).__init__()
        self.code_transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim,
                                            model_config[
            "attention_dim"], model_config["num_layers"], model_config["ff_dim"],
            model_config["num_heads"], model_config["dropout_rate"])
        # self.comment_transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim,
        # 									   model_config[
        # 										   "attention_dim"], model_config["num_layers"],
        # 									   model_config["ff_dim"],
        # 									   model_config["num_heads"], model_config["dropout_rate"])

        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(256, activation="relu")
        self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, comment_indices, comment_masks, code_indices, code_masks):
        # Set up masks for our two transformers
        comment_self_masks = tf.reshape(
            comment_masks, [comment_masks.shape[0], 1, 1, comment_masks.shape[1]])
        code_self_masks = tf.reshape(
            code_masks, [code_masks.shape[0], 1, 1, code_masks.shape[1]])
        code_key_masks = code_self_masks * \
            tf.reshape(comment_masks, [
                       comment_masks.shape[0], 1, comment_masks.shape[1], 1])

        # Compute code self-attention states
        code_states = self.code_transformer(
            code_indices, masks=code_self_masks)

        # comment_states = self.code_transformer(comment_indices, masks=comment_self_masks)
        # comment_states = tf.reduce_max(comment_states, 1)

        # # Compute comment self+code-attention states
        comment_states = self.code_transformer(
            comment_indices, masks=comment_self_masks, key_states=code_states, key_masks=code_key_masks)
        comment_states = tf.reduce_max(comment_states, 1)

        code_states = tf.reduce_max(code_states, 1)
        distance = euclidean_distance(code_states, comment_states)
        states = tf.concat([comment_states, code_states, distance,
                            comment_states - code_states, comment_states + code_states], axis=-1)
        # # Max-pool states and project to classify
        states = self.fc2(self.fc1(states))
        preds = self.classify(states)
        preds = tf.squeeze(preds, -1)
        return preds

    def attention_layer(self, model_config):
        num_layers = 3
        self.self_attention = [
            AttentionLayer(attention_dim=model_config["attention_dim"], num_heads=model_config["num_heads"],
                           hidden_dim=model_config["hidden_dim"])
            for _ in range(num_layers)]
        self.layernorm1 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.layernorm2 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.ff_1 = [tf.keras.layers.Dense(
            model_config["ff_dim"], activation="relu") for _ in range(num_layers)]
        self.ff_2 = [tf.keras.layers.Dense(
            model_config["hidden_dim"]) for _ in range(num_layers)]


class SimilarityTransformerModel(tf.keras.Model):
    def __init__(self, model_config, token_vocab_dim):
        super(SimilarityTransformerModel, self).__init__()

        self.comment_transformer = Transformer(model_config["embed_dim"], model_config["hidden_dim"], token_vocab_dim,
                                               model_config["attention_dim"], model_config["num_layers"],
                                               model_config["ff_dim"],
                                               model_config["num_heads"], model_config["dropout_rate"])
        num_layers = 3
        self.self_attention = [
            AttentionLayer(attention_dim=model_config["attention_dim"], num_heads=model_config["num_heads"],
                           hidden_dim=model_config["hidden_dim"])
            for _ in range(num_layers)]
        self.layernorm1 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.layernorm2 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.ff_1 = [tf.keras.layers.Dense(
            model_config["ff_dim"], activation="relu") for _ in range(num_layers)]
        self.ff_2 = [tf.keras.layers.Dense(
            model_config["hidden_dim"]) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(
            model_config["hidden_dim"], activation='relu')
        self.classify = tf.keras.layers.Dense(4, activation='sigmoid')

    def call(self, b_comment_indices, b_comment_masks, a_comment_indices, a_comment_masks, b_code_indices, b_code_masks,
             a_code_indices, a_code_masks):
        b_code_self_masks = tf.reshape(
            b_code_masks, [b_code_masks.shape[0], 1, 1, b_code_masks.shape[1]])
        a_code_self_masks = tf.reshape(
            a_code_masks, [a_code_masks.shape[0], 1, 1, a_code_masks.shape[1]])

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

        b_code_states = self.comment_transformer(b_code_indices,
                                                 masks=b_code_self_masks)  # [batch, seq len, hidden dim]
        a_code_states = self.comment_transformer(a_code_indices,
                                                 masks=a_code_self_masks)  # [batch, seq len, hidden dim]

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

        states = tf.stack([b_b_states, b_a_states, a_b_states,
                           a_a_states], axis=1)  # [B, 4, hidden dim]
        for ix, attention in enumerate(self.self_attention):
            attn_out = attention(states)
            out1 = self.layernorm1[ix](states + attn_out)
            ffn_out = self.ff_2[ix](self.ff_1[ix](out1))
            states = self.layernorm2[ix](out1 + ffn_out)

        states = tf.reshape(states, shape=[
                            states.shape[0], states.shape[-1] * states.shape[1]])  # [B, 4 * hidden dim]
        preds = self.classify(states)  # [B, 4]
        preds = tf.reshape(preds, shape=[-1])  # flatten into a B x 4 vector
        return preds


class SharedTransformerModel(TransformerModel):

    def __init__(self, model_config, token_vocab_dim):
        super(SharedTransformerModel, self).__init__(
            model_config, token_vocab_dim)

        self.attention_layer(model_config)

    def call(self, b_comment_indices, b_comment_masks, a_comment_indices, a_comment_masks, code_indices, code_masks,
             training=True):
        # Set up masks for our three transformers
        code_self_masks = tf.reshape(
            code_masks, [code_masks.shape[0], 1, 1, code_masks.shape[1]])

        b_code_key_masks = code_self_masks * tf.reshape(b_comment_masks,
                                                        [b_comment_masks.shape[0], 1, b_comment_masks.shape[1], 1])
        a_code_key_masks = code_self_masks * tf.reshape(a_comment_masks,
                                                        [a_comment_masks.shape[0], 1, a_comment_masks.shape[1], 1])

        b_comment_self_masks = tf.reshape(
            b_comment_masks, [b_comment_masks.shape[0], 1, 1, b_comment_masks.shape[1]])
        a_comment_self_masks = tf.reshape(
            a_comment_masks, [a_comment_masks.shape[0], 1, 1, a_comment_masks.shape[1]])

        # Compute code self-attention states
        code_states = self.code_transformer(
            code_indices, masks=code_self_masks)

        # Compute before comment self+code-attention states
        b_states = self.comment_transformer(
            b_comment_indices, masks=b_comment_self_masks, key_states=code_states, key_masks=b_code_key_masks)
        b_states = tf.reduce_max(b_states, axis=1)

        a_states = self.comment_transformer(
            a_comment_indices, masks=a_comment_self_masks, key_states=code_states, key_masks=a_code_key_masks)
        a_states = tf.reduce_max(a_states, axis=1)

        states = tf.stack([b_states, a_states], axis=1)  # [B, 2, H]
        for ix, attention in enumerate(self.self_attention):
            attn_out = attention(states)
            out1 = self.layernorm1[ix](states + attn_out)
            ffn_out = self.ff_2[ix](self.ff_1[ix](out1))
            states = self.layernorm2[ix](out1 + ffn_out)

        # prediction
        states = tf.reshape(states, shape=[
                            states.shape[0], states.shape[-1] * states.shape[1]])  # [B, 2 * hidden dim]
        preds = self.classify(states)
        preds = tf.squeeze(preds, -1)
        return preds


class BaselineModel(tf.keras.layers.Layer):

    def __init__(self, model_config, token_vocab_dim):
        super(BaselineModel, self).__init__()
        random_init = tf.random_normal_initializer(
            stddev=model_config["hidden_dim"] ** -0.5)
        self.embed = tf.keras.layers.Embedding(
            token_vocab_dim, model_config["embed_dim"])

        self.lstm_layers = [tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(model_config["hidden_dim"], dropout=model_config["dropout_rate"],
                                recurrent_dropout=model_config["dropout_rate"], return_sequences=True))
                            for _ in range(model_config["num_layers"])]
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(model_config["hidden_dim"], dropout=model_config["dropout_rate"],
                                recurrent_dropout=model_config["dropout_rate"]))

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(model_config["dropout_rate"])
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(model_config["dropout_rate"])

        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.classify = tf.keras.layers.Dense(1, activation="sigmoid")

    def rnn_layers(self, states):
        for ix in range(len(self.lstm_layers)):
            states = self.lstm_layers[ix](states)
        return self.lstm(states)

    def call(self, comment_indices, comment_masks, code_indices, code_masks, leaks_indices, leaks_masks, training=True):
        comment_em = self.embed(comment_indices)
        code_em = self.embed(code_indices)
        # leaks_em = self.embed(leaks_indices)

        code_states = self.rnn_layers(code_em)  # [B, 2 * H]
        comment_states = self.rnn_layers(comment_em)  # [B, 2 * H]
        leaks_states = tf.keras.layers.Dense(
            256, activation="relu")(leaks_indices)

        states = tf.concat(
            [code_states, comment_states, leaks_states], axis=-1)
        states = self.batch_norm1(states)
        states = self.dropout1(states, training=training)
        states = self.fc1(states)
        states = self.batch_norm2(states)
        states = self.dropout2(states, training=training)
        preds = self.classify(states)  # [B, 1]
        preds = tf.squeeze(preds, -1)  # [B]
        return preds


def log_and_print(msg):
    print(msg)
    logfile = os.path.join("../data", config["training"]["model"] + ".log")
    with open(logfile, 'a+') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    main()
