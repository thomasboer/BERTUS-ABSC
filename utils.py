#!/usr/bin/env python
# encoding: utf-8

# Utility methods.
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

import numpy as np
from config import *
from sklearn.preprocessing import OneHotEncoder


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    # here the third parameter n_iter is actually represent number of epoch, not number of iteration (=number of mini-batch = number of parameter updating)
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param length:
    :param batch_size:
    :param n_iter:
    :param is_shuffle:
    :return:
    """
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    Method obtained from Trusca et al. (2020), original docstring below.

    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print('\nload word-id mapping done!\n')
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param w2v_file: this is the location of the bert embedding obtained from getBert method.
    This txt file contains 768 dim numerical vector for each word in the input sentences.
    :param embedding_dim: 768
    :param is_skip:
    :return 1: word_dict = train_word_id_mapping
    :return 2: w2v = train_w2v
    """
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []  # declare an empty list.
    word_dict = dict()
    # [0,0,...,0] represent absent words.
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v_sum = np.sum(w2v, axis=0, dtype=np.float32)
    div = np.divide(w2v_sum, cnt, dtype=np.float32)
    w2v = np.row_stack((w2v, div))
    word_dict['$t$'] = (cnt + 1)
    return word_dict, w2v  # word_dict is dictionary datatype with length = number of words in the whole sentences in one domain / w2v is a np


def change_y_to_onehot(y, pos_neu_neg=True):
    """
    Method adapted from Trusca et al. (2020), no original docstring provided.

    :param y: vector of polarities
    :param pos_neu_neg: True if three possible polarities (positive, neutral and negative)
    :return:
    """
    from collections import Counter
    count = Counter(y)
    if FLAGS.writable == 1:
        with open(FLAGS.results_file_nomask, "a") as results:
            results.write("Positive: " + str(count['1']) + ", Neutral: " + str(
                count['0']) + ", Negative: " + str(count['-1']) + ", Total: " + str(sum(count.values())) + "\n")
    print("Polarity count:", count)
    if pos_neu_neg:
        class_set = {'1', '0', '-1'}
    else:
        class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print("Polarity mapping:", y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32), y_onehot_mapping


def change_y_to_onehot_keep(y, y_onehot_mapping, pos_neu_neg=True):
    """
    Method adapted from Trusca et al. (2020), no original docstring provided.

    :param y: vector of polarities
    :param y_onehot_mapping: one-hot mapping to keep
    :param pos_neu_neg: True if three possible polarities (positive, neutral and negative)
    :return:
    """
    from collections import Counter
    count = Counter(y)
    if FLAGS.writable == 1:
        with open(FLAGS.results_file, "a") as results:
            results.write("Positive: " + str(count['1']) + ", Neutral: " + str(
                count['0']) + ", Negative: " + str(count['-1']) + ", Total: " + str(sum(count.values())) + "\n")
    print("Polarity count:", count)
    if pos_neu_neg:
        class_set = {'1', '0', '-1'}
    else:
        class_set = set(y)
    n_class = len(class_set)
    print("Polarity mapping:", y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32), y_onehot_mapping


def load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8',
                        pos_neu_neg=True):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('Load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        # Targets.
        words = lines[i + 1].lower().split()
        target = words

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        length = min(len(target_word), target_len)
        tar_len.append(length)
        target_words.append(target_word[:length] + [0] * (target_len - length))

        # Sentiment.
        y.append(lines[i + 2].strip().split()[0])

        # Left and right context.
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            tmp = words_r
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y
    y, y_onehot_mapping = change_y_to_onehot(y, pos_neu_neg=pos_neu_neg)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':  # for both training and test type_ is TC
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(all_sent), np.asarray(
            all_target), np.asarray(all_y), y_onehot_mapping
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def load_inputs_twitter_keep(input_file, y_onehot_mapping, word_id_file, sentence_len, type_='', is_r=True,
                             target_len=10, encoding='utf8', pos_neu_neg=True):
    """
    Method adapted from Trusca et al. (2020), no original docstring provided.

    :param input_file:
    :param y_onehot_mapping: one-hot mapping to keep
    :param word_id_file:
    :param sentence_len:
    :param type_:
    :param is_r:
    :param target_len:
    :param encoding:
    :param pos_neu_neg: True if three possible polarities (positive, neutral and negative)
    :return:
    """
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('Load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    # Read in txt file.
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        # Targets.
        words = lines[i + 1].lower().split()
        target = words

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # Sentiment.
        y.append(lines[i + 2].strip().split()[0])

        # Left and right context.
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            tmp = words_r
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y
    y, y_onehot_mapping = change_y_to_onehot_keep(y, y_onehot_mapping, pos_neu_neg=pos_neu_neg)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(all_sent), np.asarray(
            all_target), np.asarray(all_y), y_onehot_mapping
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def load_inputs_bertmasker(input_file, word_id_file, sentence_len, type_='', encoding='utf8', domain=''):
    '''
    actual files inserted (train data ver.)
    input_file: train_path (FLAGS.train_path = 768_restaurantDemo_2014_BERT)
    word_id_file: train_word_id_mapping (a dictionary instance. ex) {'but_0': 1, 'the_0': 2, ...}
    sentence_len: FLAGS.max_sentence_len
    type_: 'TC'
    domain: origin or target (it determines the value of the y) ex) 0 if origin, 1 if target
    '''
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('Load word-to-id done!')
    x_full = []
    sen_len_full = []
    y_domain = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines),
                   3):  # to only read the actual review sentences, not target word line and sentiment label -1/0/1
        words = lines[i].lower().split()
        target_words = lines[i + 1].lower().split()
        sentence = words
        words_full = []
        flag = True

        # code for domain y
        # y=0 if it is source domain y=1 if it is target domain
        if domain == 'source':
            y_domain.append([1.0, 0.0])
        else:
            y_domain.append([0.0, 1.0])

        for word in words:
            if word == '$t$':
                for target_word in target_words:
                    words_full.append(word_to_id[target_word])
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_full.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_full.append(word_to_id[word])
        words_full = words_full[:sentence_len]
        sen_len_full.append(len(words_full))
        x_full.append(words_full + [0] * (sentence_len - len(words_full)))

    y_nparray = np.asarray(y_domain).astype(np.float32)
    # print('y_nparray: ', y_nparray)

    return np.asarray(x_full), np.asarray(sen_len_full), y_nparray


def load_inputs_attentionmasker(input_file, word_id_file, sentence_len, type_='', encoding='utf8', domain=''):
    '''
    actual files inserted (train data ver.)
    input_file: train_path (FLAGS.train_path = 768_restaurantDemo_2014_BERT)
    word_id_file: train_word_id_mapping (a dictionary instance. ex) {'but_0': 1, 'the_0': 2, ...}
    sentence_len: FLAGS.max_sentence_len
    type_: 'TC'
    domain: origin or target (it determines the value of the y) ex) 0 if origin, 1 if target
    '''
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('Load word-to-id done!')
    x_full = []
    sen_len_full = []
    y_domain = []
    y_sentiment = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines),
                   3):  # to only read the actual review sentences, not target word line and sentiment label -1/0/1
        words = lines[i].lower().split()
        target_words = lines[i + 1].lower().split()
        sentence = words
        words_full = []
        flag = True
        y_sentiment.append(lines[i + 2])
        # code for domain y
        # y=0 if it is source domain y=1 if it is target domain
        if domain == 'source':
            y_domain.append([0.0])
        elif domain == 'target':
            y_domain.append([1.0])
        else:
            y_domain.append([2.0])

        for word in words:
            if word == '$t$':
                for target_word in target_words:
                    words_full.append(word_to_id[target_word])
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_full.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_full.append(word_to_id[word])
        words_full = words_full[:sentence_len]
        sen_len_full.append(len(words_full))
        x_full.append(words_full + [0] * (sentence_len - len(words_full)))

    y_nparray = np.asarray(y_domain).astype(np.float32)
    y_sentiment_array = np.asarray(y_sentiment).astype(np.float32)
    # print('y_nparray: ', y_nparray)

    return np.asarray(x_full), np.asarray(sen_len_full), y_nparray, y_sentiment_array


def load_inputs_cabasc(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.
    NOTE. Not used in current adaptation.

    :param input_file:
    :param word_id_file:
    :param sentence_len:
    :param type_:
    :param is_r:
    :param target_len:
    :param encoding:
    :return:
    """
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    sent_short_final, sent_final = [], []
    target_words = []
    tar_len = []
    mult_mask = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words = lines[i + 1].lower().split()

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        y.append(lines[i + 2].strip().split()[0])

        words = lines[i].lower().split()
        words_l, words_r, sent_short, sent = [], [], [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':

            mult = [1] * sentence_len
            mult[len(words_l):len(words_l) + l] = [0.5] * l
            mult_mask.append(mult)

            sent_short.extend(words_l + target_word + words_r)
            words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r[:0] = target_word
            words_r = words_r[:sentence_len]
            sen_len_r.append(len(words_r))
            x_r.append([0] * (sentence_len - len(words_r)) + words_r)
            tmp = words_l
            if is_r:
                tmp.reverse()
            sen_len.append(len(tmp))
            x.append([0] * (sentence_len - len(tmp)) + tmp)
            sent_short_final.append(sent_short)
            sent_final.append(sent_short + [0] * (sentence_len - len(sent_short)))
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
        if i == 0:
            print(
                'words left:{} \n length left: {} \n words right: {}\n length left: {}\n target: {}\n target length:{} \n sentiment: {}\n sentence:{}\n mask:{}'.format(
                    x,
                    sen_len,
                    x_r,
                    sen_len_r,
                    target_words,
                    tar_len,
                    y,
                    sent_final,
                    mult_mask
                ))

    y, _ = change_y_to_onehot(y)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(sent_short_final), np.asarray(
            sent_final), np.asarray(mult_mask)
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def mask_word_index(original, masker):
    row_index = 0
    for row in original:
        col_index = 0
        for cell in row:
            if cell in masker[row_index, :]:
                original[row_index][col_index] = 0.
            col_index += 1
        row_index += 1
