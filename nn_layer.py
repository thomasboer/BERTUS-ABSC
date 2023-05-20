#!/usr/bin/env python
# encoding: utf-8

# Definitions for layers in a neural network.
# BERTUS
# Adapted from DIWS
# https://github.com/ejoone/DIWS-ABSC/tree/main
#
# Adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
# Deep Contextual Word Embeddings and Hierarchical Attention. In: Bielikova M., Mikkonen T., Pautasso C. (eds) Web
# Engineering. ICWE 2020. Lecture Notes in Computer Science, vol 12128. Springer, Cham.
# https://doi.org/10.1007/978-3-030-50578-3_25

import numpy as np
import tensorflow as tf
from config import *


def dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.
    NOTE. Not used in current adaptation (for CABASC model).

    :param cell:
    :param inputs:
    :param n_hidden:
    :param length:
    :param max_len:
    :param scope_name:
    :param out_type:
    :return:
    """
    outputs, state = tf.nn.dynamic_rnn(
        cell(n_hidden),
        inputs=inputs,
        dtype=tf.float32,
        scope=scope_name
    )  # outputs -> batch_size * max_len * n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)
    mask = tf.reverse(tf.cast(tf.sequence_mask(length, max_len), tf.float32), [1])
    mask_tiled = tf.tile(mask, [1, n_hidden])
    mask_3d = tf.reshape(mask_tiled, tf.shape(outputs))
    return tf.multiply(outputs, mask_3d)


def bi_dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param cell:
    :param inputs:
    :param n_hidden:
    :param length:
    :param max_len:
    :param scope_name:
    :param out_type:
    :return:
    """
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    if out_type == 'last':
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    else:
        outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def reduce_mean_with_len(inputs, length):
    """
    Method obtained from Trusca et al. (2020), original docstring below.

    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    return inputs


def softmax_layer(inputs, n_hidden, random_base, keep_prob, l2_reg, n_class, scope_name='1'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param inputs:
    :param n_hidden:
    :param random_base:
    :param keep_prob:
    :param l2_reg:
    :param n_class:
    :param scope_name:
    :return:
    """
    w = tf.get_variable(
        name='softmax_w' + scope_name,
        shape=[n_hidden, n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b' + scope_name,
        shape=[n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / n_class)),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        predict = tf.matmul(outputs, w) + b  # output of MLP(.)?
        predict = tf.nn.softmax(predict)
    return predict


def MLP_layer(inputs, embedding_dim, random_base, keep_prob, l2_reg, n_class, scope_name='1'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param inputs:
    :param n_hidden:
    :param random_base:
    :param keep_prob:
    :param l2_reg:
    :param n_class:
    :param scope_name:
    :return:
    """
    with tf.variable_scope('softmax_w' + scope_name, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable(
            name='softmax_w' + scope_name,
            shape=[embedding_dim, n_class],
            initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    with tf.variable_scope('softmax_b' + scope_name, reuse=tf.AUTO_REUSE) as scope:
        b = tf.get_variable(
            name='softmax_b' + scope_name,
            shape=[n_class],
            initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / n_class)),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        predict = tf.matmul(outputs, w) + b  # output of MLP(.)?

        predict = tf.nn.softmax(predict)

    return predict


def MLP_layer_z(inputs, embedding_dim, random_base, keep_prob, l2_reg, n_class, scope_name='1'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param inputs:
    :param n_hidden:
    :param random_base:
    :param keep_prob:
    :param l2_reg:
    :param n_class:
    :param scope_name:
    :return:
    """
    batch_size, a, b = inputs.get_shape()
    with tf.variable_scope('softmaxz_w' + scope_name, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable(
            name='softmaxz_w' + scope_name,
            shape=[batch_size, embedding_dim, n_class],
            initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    with tf.variable_scope('softmaxz_b' + scope_name, reuse=tf.AUTO_REUSE) as scope:
        b = tf.get_variable(
            name='softmaxz_b' + scope_name,
            shape=[n_class],
            initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / n_class)),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    with tf.name_scope('softmaxz'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        predict = tf.matmul(outputs, w) + b  # output of MLP(.)?
    return predict


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=True):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y
