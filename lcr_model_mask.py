#!/usr/bin/env python
# encoding: utf-8

# Regular LCR-Rot-hop++ method that can process transformed input sentences by BERTUS.
#
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

import os

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from utils import load_w2v, batch_index, load_inputs_twitter, load_inputs_twitter_keep, load_inputs_bertmasker, \
    mask_word_index

sys.path.append(os.getcwd())
tf.set_random_seed(1)


def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):
    # Structure of LCR-Rot-hop++ neural network.
    print('I am lcr_rot_hop_plusplus.')
    cell = tf.contrib.rnn.LSTMCell
    # Left Bi-LSTM.
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    # print('input_fw in lcr_rot model', input_fw)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    # print('hiddens_l in lcr_rot model', hiddens_l)

    # Right Bi-LSTM.
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')

    # Target Bi-LSTM.
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # Left context attention layer.
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l')
    # print('att_l in lcr_Rot: ', att_l)
    outputs_l_init = tf.matmul(att_l, hiddens_l)
    outputs_l = tf.squeeze(outputs_l_init)

    # Right context attention layer.
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r')
    outputs_r_init = tf.matmul(att_r, hiddens_r)
    outputs_r = tf.squeeze(outputs_r_init)

    # Left-aware target attention layer.
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'tl')
    outputs_t_l_init = tf.matmul(att_t_l, hiddens_t)

    # Right-aware target attention layer.
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'tr')
    outputs_t_r_init = tf.matmul(att_t_r, hiddens_t)

    # Context and target hierarchical attention layers.
    outputs_init_context = tf.concat([outputs_l_init, outputs_r_init], 1)
    outputs_init_target = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                      FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                     FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    # Add two more hops.
    for i in range(2):
        # Left context attention layer.
        att_l = bilinear_attention_layer(hiddens_l, outputs_t_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'l' + str(i))
        outputs_l_init = tf.matmul(att_l, hiddens_l)
        outputs_l = tf.squeeze(outputs_l_init)

        # Right context attention layer.
        att_r = bilinear_attention_layer(hiddens_r, outputs_t_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'r' + str(i))
        outputs_r_init = tf.matmul(att_r, hiddens_r)
        outputs_r = tf.squeeze(outputs_r_init)

        # Left-aware target attention layer.
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'tl' + str(i))
        outputs_t_l_init = tf.matmul(att_t_l, hiddens_t)

        # Right-aware target attention layer.
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'tr' + str(i))
        outputs_t_r_init = tf.matmul(att_t_r, hiddens_t)

        # Context and target hierarchical attention layers.
        outputs_init_context = tf.concat([outputs_l_init, outputs_r_init], 1)
        outputs_init_target = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                          FLAGS.random_base, 'fin1' + str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                         FLAGS.random_base, 'fin2' + str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    # print('outputs l',outputs_l)
    # MLP for sentiment classification.
    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)

    prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)
    return prob, att_l, att_r, att_t_l, att_t_r, hiddens_l
    # att_l with sahpe (?,1,?) here is alpha_fw


def main(train_path, test_path, mask_source, mask_target, learning_rate=0.09, keep_prob=0.4, momentum=0.85, l2=0.0001):
    # Runs the regular LCR-Rot-hop++ method.
    # print_config()
    train_acc_list = np.empty(shape=(0), dtype=float)
    test_acc_list = np.empty(shape=(0), dtype=float)
    with tf.device('/cpu:0'):
        # In this section, we define the "calculations" that we will perform using sess.run with feed_dict (actual filled value)
        # Separate train and test embeddings for cross-domain classification.
        if FLAGS.train_embedding == FLAGS.test_embedding:
            train_word_id_mapping, train_w2v = load_w2v(FLAGS.train_embedding, FLAGS.embedding_dim)
            train_word_embedding = tf.constant(train_w2v, dtype=np.float32, name='train_word_embedding')
            test_word_id_mapping = train_word_id_mapping
        else:
            train_word_id_mapping, train_w2v = load_w2v(FLAGS.train_embedding, FLAGS.embedding_dim)
            train_word_embedding = tf.constant(train_w2v, dtype=np.float32, name='train_word_embedding')
            test_word_id_mapping, test_w2v = load_w2v(FLAGS.test_embedding, FLAGS.embedding_dim)
            test_word_embedding = tf.constant(test_w2v, dtype=np.float32, name='test_word_embedding')

        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len = tf.placeholder(tf.int32, None)

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw = tf.placeholder(tf.int32, [None])

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])

            inputs_fw_holder_2 = tf.placeholder(tf.float32, [None, FLAGS.max_sentence_len, FLAGS.embedding_dim])
            inputs_bw_holder_2 = tf.placeholder(tf.float32, [None, FLAGS.max_sentence_len, FLAGS.embedding_dim])
            target_holder_2 = tf.placeholder(tf.float32, [None, FLAGS.max_target_len, FLAGS.embedding_dim])


        prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, hidden_lstm = lcr_rot(inputs_fw_holder_2, inputs_bw_holder_2,
                                                                              sen_len, sen_len_bw, target_holder_2,
                                                                              tar_len, keep_prob1, keep_prob2, l2,
                                                                              'all')
        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss,
                                                                                                        global_step=global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize weights and biases.
        sess.run(tf.global_variables_initializer())
        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _, y_onehot_mapping = load_inputs_twitter(
            train_path, train_word_id_mapping, FLAGS.max_sentence_len, 'TC', is_r, FLAGS.max_target_len)
        # print(tr_x)
        mask_word_index(tr_x, mask_source)
        mask_word_index(tr_target_word, mask_source)
        mask_word_index(tr_x_bw, mask_source)
        print(tr_x)

        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _, _ = load_inputs_twitter_keep(
            test_path, y_onehot_mapping, test_word_id_mapping, FLAGS.max_sentence_len, 'TC', is_r, FLAGS.max_target_len,
            pos_neu_neg=True)
        mask_word_index(te_x, mask_target)
        mask_word_index(te_target_word, mask_target)
        mask_word_index(te_x_bw, mask_target)

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, batch_target, batch_tl, batch_size, kp1, kp2,
                           isTrain=True, is_shuffle=True):
            if isTrain:
                inputs_fw_2 = tf.nn.embedding_lookup(train_word_embedding, x_f)
                inputs_fw_2_array = inputs_fw_2.eval(session=tf.Session())
                inputs_bw_2 = tf.nn.embedding_lookup(train_word_embedding, x_b)
                inputs_bw_2_array = inputs_bw_2.eval(session=tf.Session())
                inputs_target_2 = tf.nn.embedding_lookup(train_word_embedding, batch_target)
                inputs_target_2_array = inputs_target_2.eval(session=tf.Session())
            else:
                inputs_fw_2 = tf.nn.embedding_lookup(test_word_embedding, x_f)
                inputs_fw_2_array = inputs_fw_2.eval(session=tf.Session())
                inputs_bw_2 = tf.nn.embedding_lookup(test_word_embedding, x_b)
                inputs_bw_2_array = inputs_bw_2.eval(session=tf.Session())
                inputs_target_2 = tf.nn.embedding_lookup(test_word_embedding, batch_target)
                inputs_target_2_array = inputs_target_2.eval(session=tf.Session())
            for index in batch_index(len(yi), batch_size, 1,
                                     is_shuffle):  # here 1 is number of iteration = epoch와 같은 의미로 쓴듯?
                feed_dict = {x: x_f[index], x_bw: x_b[index], y: yi[index], sen_len: sen_len_f[index],
                             sen_len_bw: sen_len_b[index], target_words: batch_target[index],
                             tar_len: batch_tl[index], keep_prob1: kp1, keep_prob2: kp2,
                             inputs_fw_holder_2: inputs_fw_2_array[index], inputs_bw_holder_2: inputs_bw_2_array[index],
                             target_holder_2: inputs_target_2_array[index]}
                yield feed_dict, len(index)

        max_acc = 0.
        min_acc = 1.

        fwlayer = []
        for i in range(FLAGS.n_iter):
            trainacc, traincnt = 0., 0
            # Train model.
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word,
                                                  tr_tar_len, FLAGS.batch_size, keep_prob, keep_prob):
                _, step, _trainacc, _layer = sess.run([optimizer, global_step, acc_num, hidden_lstm], feed_dict=train)

                trainacc += _trainacc
                traincnt += numtrain
                layer = np.asarray(_layer)
            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []


            # Test model.
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len,
                                            2000, 1.0, 1.0, isTrain=False, is_shuffle=False):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run(
                        [loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r], feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                acc += _acc
                cost += _loss * num
                cnt += num

            print('Total samples={}, correct predictions={}'.format(cnt, acc))
            trainacc = trainacc / traincnt
            acc = acc / cnt
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}'.format(i, cost, trainacc, acc))
            train_acc_list = np.append(train_acc_list, trainacc)
            test_acc_list = np.append(test_acc_list, acc)

            if acc > max_acc:
                max_acc = acc
                test_acc_of_max_acc = trainacc
            if acc < min_acc:
                min_acc = acc
                test_acc_of_min_acc = trainacc

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(
                    "---\nLCR-Rot-Hop++ Attention Masker. Mean Train accuracy: {:.6f}, Train accuracy standard deviation: {:.6f}\n".format(
                        np.average(train_acc_list), np.std(train_acc_list)))
                results.write(
                    "LCR-Rot-Hop++ Domain Attention Masker. Mean Test accuracy: {:.6f}, Test accuracy standard deviation: {:.6f}\n".format(
                        np.average(test_acc_list), np.std(test_acc_list)))
                results.write("Maximum (highest) Test accuracy: {:.6f}\n".format(max_acc))
                results.write("corresponding training accuracy to the Maximum (highest) Test accuracy: {:.6f}\n".format(
                    test_acc_of_max_acc))
                results.write("Minimum (lowest) Test accuracy: {:.6f}\n".format(min_acc))
                results.write("corresponding training accuracy to the Minimum (lowest) Test accuracy: {:.6f}\n".format(
                    test_acc_of_min_acc))
        print('Optimization Finished! Test accuracy={}\n'.format(acc))

        # Record accuracy by polarity.
        FLAGS.pos = y_onehot_mapping['1']
        FLAGS.neu = y_onehot_mapping['0']
        FLAGS.neg = y_onehot_mapping['-1']
        pos_count = 0
        neg_count = 0
        neu_count = 0
        pos_correct = 0
        neg_correct = 0
        neu_correct = 0
        for i in range(0, len(ty)):
            if ty[i] == FLAGS.pos:
                # Positive sentiment.
                pos_count += 1
                if py[i] == FLAGS.pos:
                    pos_correct += 1
            elif ty[i] == FLAGS.neu:
                # Neutral sentiment.
                neu_count += 1
                if py[i] == FLAGS.neu:
                    neu_correct += 1
            else:
                # Negative sentiment.
                neg_count += 1
                if py[i] == FLAGS.neg:
                    neg_correct += 1
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Test results.\n")
                results.write(
                    "Positive. Correct: {}, Incorrect: {}, Total: {}\n".format(pos_correct, pos_count - pos_correct,
                                                                               pos_count))
                results.write(
                    "Neutral. Correct: {}, Incorrect: {}, Total: {}\n".format(neu_correct, neu_count - neu_correct,
                                                                              neu_count))
                results.write(
                    "Negative. Correct: {}, Incorrect: {}, Total: {}\n---\n".format(neg_correct,
                                                                                    neg_count - neg_correct,
                                                                                    neg_count))

        return acc, np.where(np.subtract(py, ty) == 0, 0, 1), fw.tolist(), bw.tolist(), tl.tolist(), tr.tolist()


if __name__ == '__main__':
    tf.app.run()
