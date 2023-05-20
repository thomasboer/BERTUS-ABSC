# BERTUS model for hyper training and testing
#
# Adapted from DIWS
# https://github.com/ejoone/DIWS-ABSC

import tensorflow as tf
import pandas as pd
import numpy as np
import random
from config import *
from utils import load_inputs_bertmasker, load_w2v, load_inputs_attentionmasker
from sklearn.model_selection import train_test_split
import flipGradientTF


reduce_size = 30
tf.set_random_seed(1)

def sample_gumbel(shape, eps=1e-20):
    """
    Sample from Gumbel distribution.
    """
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature):
    """
    Sample from the Gumbel-Softmax distribution and return the sample and its corresponding
    one-hot encoding.
    """
    y = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(y / temperature)
    return y


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

hidden_size = 100
num_classes = 2

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, domain, units, **kwargs):
        self.units = units
        self.domain = domain
        self.reverse_domain = tf.constant(1.) - self.domain
        super(MyDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='source',
                                 shape=(1, 1, self.units),
                                 initializer='zeros',
                                 trainable=True)
        self.t = self.add_weight(name='target',
                                 shape=(1, 1, self.units),
                                 initializer='zeros',
                                 trainable=True)
        super(MyDenseLayer, self).build(input_shape)

    def call(self, inputs):
        # return tf.concat([inputs, tf.tile(self.w,[1,80,1])],axis=-1)
        return tf.tile(self.w,[1,1,1])*tf.tile(tf.expand_dims(self.domain,-1),[1,1,200]) + tf.tile(self.t,[1,1,1])*tf.tile(tf.expand_dims(self.reverse_domain,-1),[1,1,200])

def build_model():
    domain_input_layer = tf.keras.layers.Input(shape=(1,),dtype=tf.float32)

    #Concatenating input embeddings with domain descriptors
    input_layer = tf.keras.layers.Input(shape=(80, FLAGS.embedding_dim))
    domain_descriptor_source = MyDenseLayer(domain_input_layer, units=200,name='domain_descriptor_source', trainable=True)(input_layer)
    domain_descriptor_source_tiled = tf.keras.layers.Lambda(tf.tile, arguments={"multiples":[1,80,1]})(domain_descriptor_source)
    result_concat_source = tf.keras.layers.Concatenate(axis=2)([input_layer, domain_descriptor_source_tiled])

    #Token masking network
    l1 = tf.keras.layers.Dense(hidden_size, activation='tanh')(result_concat_source)
    l2 = tf.keras.layers.Dense(num_classes,name='similarity_scores')(l1)
    y = tf.keras.layers.Lambda(gumbel_softmax, arguments={'temperature': 0.1},name='mask_decisions')(l2)

    prepare_for_concat = tf.keras.layers.Lambda(tf.keras.backend.expand_dims, arguments={'axis': 2})(input_layer)
    expand_y_pred = tf.keras.layers.Lambda(tf.keras.backend.expand_dims, arguments={'axis': -1})(y)
    zeros_tensor = tf.keras.layers.Lambda(tf.keras.backend.zeros_like)(prepare_for_concat)
    concat_mask_with_input = tf.keras.layers.Concatenate(axis=2)([prepare_for_concat,zeros_tensor])
    masked_embeds = tf.keras.layers.multiply([expand_y_pred, concat_mask_with_input])
    result_embeds = tf.keras.layers.Lambda(tf.keras.backend.sum,arguments={'axis': 2})(masked_embeds)

    #Shared prepped tensors
    concat_mask_with_input_shared = tf.keras.layers.Concatenate(axis=2)([zeros_tensor,prepare_for_concat])
    masked_embeds_shared = tf.keras.layers.multiply([expand_y_pred, concat_mask_with_input_shared])
    result_embeds_shared = tf.keras.layers.Lambda(tf.keras.backend.sum, arguments={'axis': 2})(masked_embeds_shared)

    mean_layer_shared = tf.keras.layers.Lambda(tf.keras.backend.mean, arguments={"axis": 1})(result_embeds_shared)
    grl = tf.keras.layers.Lambda(grad_reverse)(mean_layer_shared)
    domain_l1_shared = tf.keras.layers.Dense(hidden_size, activation='tanh')(grl)
    y_domain_shared = tf.keras.layers.Dense(2, activation='softmax', name='domain_class_shared')(domain_l1_shared)

    #Domain Classification
    mean_layer = tf.keras.layers.Lambda(tf.keras.backend.mean, arguments={"axis":1})(result_embeds)
    domain_l1 = tf.keras.layers.Dense(hidden_size, activation='tanh')(mean_layer)
    y_domain = tf.keras.layers.Dense(2, activation='softmax',name='domain_class')(domain_l1)

    #Sentiment Classification
    sent_l1 = tf.keras.layers.Dense(hidden_size, activation='tanh')(mean_layer_shared)
    y_sentiment = tf.keras.layers.Dense(1, activation='sigmoid', name='sentiment_class')(sent_l1)
    return tf.keras.Model(inputs=[input_layer,domain_input_layer], outputs=[y_domain,y_domain_shared,y_sentiment])


def load_data(source_path, target_path):
    source_word_id_mapping, source_w2v = load_w2v(FLAGS.source_embedding, FLAGS.embedding_dim)
    target_word_id_mapping, target_w2v = load_w2v(FLAGS.target_embedding, FLAGS.embedding_dim)
    source_word_embedding = tf.constant(source_w2v, name='source_word_embedding')
    target_word_embedding = tf.constant(target_w2v, name='target_word_embedding')
    word_embedding = tf.concat([source_word_embedding, target_word_embedding], axis=0)

    source_x, source_sen_len, source_y, source_y_sentiment = load_inputs_attentionmasker(source_path, source_word_id_mapping,
                                                                     FLAGS.max_sentence_len, 'TC', domain='source')
    target_x, target_sen_len, target_y, target_y_sentiment = load_inputs_attentionmasker(target_path, target_word_id_mapping,
                                                                     FLAGS.max_sentence_len, 'TC', domain='target')

    num_word_emb_source, _ = source_word_embedding.get_shape()
    target_x_v2 = target_x + num_word_emb_source
    x_concat = np.concatenate((source_x, target_x_v2), axis=0)
    sen_len_concat = np.concatenate((source_sen_len, target_sen_len), axis=0)
    y_concat = np.concatenate((source_y, target_y), axis=0)
    y_sentiment_concat = np.concatenate((source_y_sentiment, target_y_sentiment), axis=0)

    x_concat_tensor = tf.convert_to_tensor(x_concat, dtype=tf.int32)

    inputs = tf.nn.embedding_lookup(word_embedding, x_concat_tensor)
    inputs_nparray = inputs.eval(session=tf.Session())

    return inputs_nparray, y_concat, y_sentiment_concat, sen_len_concat, source_sen_len, target_sen_len, len(source_y), len(target_y)

def main(source_path_train, target_path_train, source_path_test, target_path_test, learning_rate, momentum,
         epochs_hyper, batch_size_hyper):
    tf.keras.backend.clear_session()
    model = build_model()
    # model.summary()

    inputs_x, inputs_y, inputs_y_sentiment, sen_len, source_sen_len, target_sen_len, numdata_source, numdata_target = load_data(
        source_path_train, target_path_train)
    inputs_y_sentiment = np.expand_dims(inputs_y_sentiment, axis=-1)
    inputs_y_sentiment = np.where(inputs_y_sentiment == -1, 0, inputs_y_sentiment)


    # Convert to integer array and flatten
    arr_int = inputs_y.astype(int).flatten()

    # Create one-hot array
    one_hot_arr = np.zeros((len(arr_int), 2))
    one_hot_arr[np.arange(len(arr_int)), arr_int] = 1.

    idx = np.random.permutation(inputs_x.shape[0])
    inputs_x_shuffled = inputs_x[idx]
    inputs_y_shuffled = inputs_y[idx]
    inputs_y_sentiment_shuffled = inputs_y_sentiment[idx]
    inputs_y_shuffled_classification = one_hot_arr[idx]


    test_x, test_y, test_y_sentiment, _, _, _, _, _ = load_data(source_path_test, target_path_test)
    test_y_sentiment = np.expand_dims(test_y_sentiment, axis=-1)
    test_y_sentiment = np.where(test_y_sentiment == -1, 0, test_y_sentiment)

    # Convert to integer array and flatten
    arr_int = test_y.astype(int).flatten()

    # Create one-hot array
    one_hot_arr = np.zeros((len(arr_int), 2))
    one_hot_arr[np.arange(len(arr_int)), arr_int] = 1.

    test_y_class = one_hot_arr

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=momentum), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit([inputs_x_shuffled,inputs_y_shuffled], [inputs_y_shuffled_classification,inputs_y_shuffled_classification,inputs_y_sentiment_shuffled], epochs=epochs_hyper, batch_size=batch_size_hyper)

    score = model.evaluate([test_x,test_y], [test_y_class,test_y_class,test_y_sentiment], batch_size=batch_size_hyper)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score[1]
