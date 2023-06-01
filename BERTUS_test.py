# BERTUS model for testing domain classification part
# BERTUS
#
# Adapted from DIWS
# https://github.com/ejoone/DIWS-ABSC/tree/main

import numpy as np
import random
from config import *
from utils import load_w2v, load_inputs_attentionmasker
from numpy.random import seed
from sklearn.model_selection import train_test_split


reduce_size = 30
tf.set_random_seed(1)
seed(5)
random.seed(1)

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
    y = tf.keras.layers.Lambda(gumbel_softmax, arguments={'temperature': 0.001},name='mask_decisions')(l2)
    slice_mask = tf.keras.layers.Lambda(tf.slice, arguments={'begin': [0,0,0],'size': [-1,-1,1]}, name='slice')(y)
    # sum_mask = tf.keras.layers.Lambda(tf.keras.backend.sum,arguments={'axis': [1]},name='sum_mask')(slice_mask)
    # check_sum = tf.keras.layers.Lambda(tf.keras.backend.greater_equal, arguments={'y': tf.constant([0.1])}, name='check_sum')(sum_mask)
    prepare_for_concat = tf.keras.layers.Lambda(tf.keras.backend.expand_dims, arguments={'axis': 2})(input_layer)
    expand_y_pred = tf.keras.layers.Lambda(tf.keras.backend.expand_dims, arguments={'axis': -1})(y)
    zeros_tensor = tf.keras.layers.Lambda(tf.keras.backend.zeros_like)(prepare_for_concat)
    concat_mask_with_input = tf.keras.layers.Concatenate(axis=2)([prepare_for_concat,zeros_tensor])
    masked_embeds = tf.keras.layers.multiply([expand_y_pred, concat_mask_with_input])
    result_embeds = tf.keras.layers.Lambda(tf.keras.backend.sum,arguments={'axis': 2},name='private_result_embed')(masked_embeds)

    # result_embeds = tf.keras.layers.multiply([input_layer, slice_mask])

    #Shared prepped tensors
    concat_mask_with_input_shared = tf.keras.layers.Concatenate(axis=2)([zeros_tensor,prepare_for_concat])
    masked_embeds_shared = tf.keras.layers.multiply([expand_y_pred, concat_mask_with_input_shared])
    result_embeds_shared = tf.keras.layers.Lambda(tf.keras.backend.sum, arguments={'axis': 2},name='shared_result_embed')(masked_embeds_shared)
    flat_layer_shared = tf.keras.layers.Flatten()(result_embeds_shared)
    # grl = flipGradientTF.GradientReversal(1)(flat_layer_shared)
    mean_layer_shared = tf.keras.layers.Lambda(tf.keras.backend.mean, arguments={"axis": 1})(result_embeds_shared)
    grl = tf.keras.layers.Lambda(grad_reverse)(mean_layer_shared)
    domain_l1_shared = tf.keras.layers.Dense(hidden_size, activation='tanh')(grl)
    y_domain_shared = tf.keras.layers.Dense(2, activation='softmax', name='domain_class_shared')(domain_l1_shared)
    # grl = tf.keras.layers.Lambda(grad_reverse)(y_domain_shared)

    #Domain Classification
    flat_layer = tf.keras.layers.Flatten()(result_embeds)
    mean_layer = tf.keras.layers.Lambda(tf.keras.backend.mean, arguments={"axis":1})(result_embeds)
    domain_l1 = tf.keras.layers.Dense(hidden_size, activation='tanh')(mean_layer)
    y_domain = tf.keras.layers.Dense(2, activation='softmax',name='domain_class')(domain_l1)
    # grl = tf.keras.layers.Lambda(grad_reverse)(y_domain)

    #Sentiment Classification
    # flat_layer = tf.keras.layers.Flatten()(mean_layer)
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

def choose_domain_descriptor():
    domain = tf.constant([1,0,0],dtype=tf.float32)
    domain_descriptor_Camera = tf.Variable(tf.random_normal_initializer()((200,)), trainable=True)
    domain_descriptor_Creative = tf.Variable(tf.random_normal_initializer()((200,)), trainable=True)
    domain_descriptor_Apex = tf.Variable(tf.random_normal_initializer()((200,)), trainable=True)
    all_domain_descriptors = tf.stack([domain_descriptor_Camera, domain_descriptor_Creative, domain_descriptor_Apex], axis=0)
    domain = tf.reshape(domain, (tf.shape(x)[0], 3))
    current_domain = tf.matmul(domain,all_domain_descriptors)
    domain_descriptor_final = tf.tile(tf.expand_dims(current_domain,1), [1,80,1], name='dom')
    return domain_descriptor_final

def main(source_path1, target_path1, learning_rate, momentum, epochs_hyper, batch_size_hyper,lambda_private,lambda_shared,lambda_sent,val_split):
    # input data should be an integreated word embedding data with source and target domains
    tf.keras.backend.clear_session()
    model = build_model()
    model.summary()

    inputs_x, inputs_y, inputs_y_sentiment, sen_len, source_sen_len, target_sen_len, numdata_source, numdata_target = load_data(
        source_path1, target_path1)
    inputs_y_sentiment = np.expand_dims(inputs_y_sentiment,axis=-1)
    inputs_y_sentiment = np.where(inputs_y_sentiment == -1, 0, inputs_y_sentiment)
    print(inputs_y_sentiment[3:15])
    print(len(target_sen_len))
    inputs_y_mask = np.squeeze(np.array([[5]*1071]),0)
    print(inputs_y_mask)


    # Convert to integer array and flatten
    arr_int = inputs_y.astype(int).flatten()

    # Create one-hot array
    one_hot_arr = np.zeros((len(arr_int), 2))
    one_hot_arr[np.arange(len(arr_int)), arr_int] = 1.

    inputs_x_train, val_inputs_x, inputs_y_train, val_inputs_y, inputs_y_classification_train, val_inputs_y_classification, inputs_y_sentiment_train, inputs_y_sentiment_val = train_test_split(inputs_x,inputs_y, one_hot_arr, inputs_y_sentiment, shuffle=True, test_size=0.2, random_state=42)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=momentum), loss={'domain_class': 'binary_crossentropy', 'domain_class_shared': 'binary_crossentropy', 'sentiment_class': 'binary_crossentropy'},
              metrics={'domain_class': 'accuracy',  'domain_class_shared': 'accuracy', 'sentiment_class': 'accuracy'}, loss_weights={'domain_class': lambda_private, 'domain_class_shared': lambda_shared, 'sentiment_class': lambda_sent})


    model.fit([inputs_x_train,inputs_y_train], [inputs_y_classification_train,inputs_y_classification_train,inputs_y_sentiment_train], epochs=epochs_hyper, batch_size=batch_size_hyper,shuffle=True)
    # model.fit(inputs_x, inputs_y, epochs=1, batch_size=10, validation_split=0.5, verbose=2)


    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=momentum), loss={'domain_class': 'binary_crossentropy', 'domain_class_shared': 'binary_crossentropy', 'sentiment_class': 'binary_crossentropy'},
              metrics={'domain_class': 'accuracy',  'domain_class_shared': 'accuracy', 'sentiment_class': 'accuracy'}, loss_weights={'domain_class': 0, 'domain_class_shared': 0, 'sentiment_class': 1})

    model.fit([inputs_x[:len(source_sen_len)],inputs_y[:len(source_sen_len)]], [one_hot_arr[:len(source_sen_len)],one_hot_arr[:len(source_sen_len)],inputs_y_sentiment[:len(source_sen_len)]], epochs=15, batch_size=batch_size_hyper,shuffle=True)

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    #
    output_data = activation_model.predict([val_inputs_x, val_inputs_y])
    print('Private part domain classification acc: ' + str(
        tf.Session().run(acc_func(val_inputs_y_classification, output_data[-3]))))
    print('Shared part domain classification acc: ' + str(
        tf.Session().run(acc_func(val_inputs_y_classification, output_data[-2]))))

    output_data = activation_model.predict([inputs_x, inputs_y])

    masked_matrix = np.squeeze(output_data[7][:,:,0])

    import pandas as pd

    loss_training = pd.DataFrame(model.history.history['loss'])
    # val_loss = pd.DataFrame(model.history.history['val_loss'])

    loss_training.to_csv('loss_training.csv')
    # val_loss.to_csv('val_loss.csv')


    #
    # plt.plot(model.history.history['loss'], label='Training Loss')
    # plt.plot(model.history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.show()

    list_count_mask = []

    for i in range(0,len(source_sen_len)):
        masked_words_in_one_sentence = np.sum(masked_matrix[i,:source_sen_len[i]])
        list_count_mask.append(masked_words_in_one_sentence/source_sen_len[i])
    print(np.average(list_count_mask))

    list_count_mask_target = []
    for i in range(0,len(target_sen_len)):
        masked_words_in_one_sentence = np.sum(masked_matrix[i+len(source_sen_len), :target_sen_len[i]])
        list_count_mask_target.append(masked_words_in_one_sentence / target_sen_len[i])
    print(np.average(list_count_mask_target))

    import pandas as pd

    target_percentages = pd.DataFrame(list_count_mask_target)
    source_percentages = pd.DataFrame(list_count_mask)

    target_percentages.to_csv('target_count.csv')
    source_percentages.to_csv('source_count.csv')
    return masked_matrix, source_sen_len, target_sen_len, numdata_source, numdata_target

