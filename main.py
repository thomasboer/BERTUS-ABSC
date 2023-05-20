# Main training and testing
#
# BERTUS
#
# Adapted from DIWS
# https://github.com/ejoone/DIWS-ABSC/tree/main
#

import nltk
import BERTUS_model
import lcr_model_mask
from config import *
from load_data import *

nltk.download('punkt')

tf.set_random_seed(1)

def main(_):
    apex_domain = ["Apex", 2004, 0.0001, 0.9, 10, 30, 0.8, 0.1, 1.0,0.2]
    camera_domain = ["Camera", 2004, 0.0001, 0.85, 15, 15, 0.8, 0.1, 1.0,0.2]
    hotel_domain = ["hotel", 2015, 0.0001, 0.99, 15, 15, 0.8, 0.1, 1.0,0.2]
    nokia_domain = ["Nokia", 2004, 0.0001, 0.95, 15, 10, 0.8, 0.1, 1,0.2]
    domains = [camera_domain,apex_domain,nokia_domain,hotel_domain]

    for domain in domains:
        main_perdomain(domain)


def main_perdomain(domain):
    set_hyper_flags(learning_rate=0.09, keep_prob=0.4, momentum=0.85, l2_reg=0.0001)
    set_other_flags(source_domain="Creative", source_year=2004, target_domain=domain[0], target_year=domain[1])
    print('main run DIWS')
    attention_final_ouput, source_sen_len, target_sen_len, numdata_source, numdata_target = BERTUS_model.main(
        FLAGS.source_path, FLAGS.target_path, domain[2], domain[3], domain[4], domain[5], domain[6], domain[7], domain[8],domain[9])

    print('current domain:', domain[0])
    set_hyper_flags(learning_rate=0.09, keep_prob=0.4, momentum=0.85, l2_reg=0.0001)
    set_other_flags(source_domain="Creative", source_year=2004, target_domain=domain[0], target_year=domain[1])
    mask_source_output, mask_target_output = get_masker(attention_final_ouput, source_sen_len, target_sen_len,
                                                        numdata_source, numdata_target)

    print(mask_target_output, mask_target_output.shape)
    print(mask_target_output[150, :], mask_target_output[1, :].shape)
    print(mask_source_output[0,:])

    print('Main Run new LCR')
    lcr_model_mask.main(FLAGS.train_path, FLAGS.test_path, mask_source_output, mask_target_output,
                        FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
    tf.reset_default_graph()


def get_masker(attention_final_main, source_sen_len, target_sen_len, numdata_source, numdata_target):
    row_index = 0
    col_index = 0
    mask_source = []
    mask_target = []
    row_ind = 0
    col_ind = 0
    past_word_ind = 0
    word_ind = 0

    # construct masking matrix
    attention_final = attention_final_main
    row_index = 0
    for row in attention_final:
        row_nonzero = row[row != 0]
        if row_nonzero.size != 0:  # if row_nonzero is not empty
            threshold_abs = 1
        else:  # if row_nonzero is empty
            threshold_abs = 1
        col_index = 0
        for cell in row:
            if cell > 0.9:
                attention_final[row_index][col_index] = 1  # value 1 corresponds to the domain specific masked words
            else:
                attention_final[row_index][col_index] = 0
            col_index += 1
        row_index += 1

    mask_source = attention_final[0:numdata_source, :]
    mask_target = attention_final[numdata_source:, :]
    row_ind = 0
    for row in mask_source:
        past_word_ind = 0
        col_ind = 0
        if row_ind == 0:
            word_ind = 1
        else:
            for k in range(row_ind):
                past_word_ind += source_sen_len[k]
            word_ind = past_word_ind + 1
        for cell in row:
            if cell == 1:
                mask_source[row_ind][col_ind] = word_ind
            else:
                mask_source[row_ind][col_ind] = 0
            col_ind += 1
            word_ind += 1
        row_ind += 1

    row_ind = 0
    for row in mask_target:
        past_word_ind = 0
        col_ind = 0
        if row_ind == 0:
            word_ind = 1
        else:
            for k in range(row_ind):
                past_word_ind += target_sen_len[k]
            word_ind = past_word_ind + 1
        for cell in row:
            if cell == 1:
                mask_target[row_ind][col_ind] = word_ind
            else:
                mask_target[row_ind][col_ind] = 0
            col_ind += 1
            word_ind += 1
        row_ind += 1

    return mask_source, mask_target


def set_hyper_flags(learning_rate, keep_prob, momentum, l2_reg):
    """
    Sets hyperparameter flags.

    :param learning_rate: learning rate hyperparameter
    :param keep_prob: keep probability hyperparameter
    :param momentum: momentum hyperparameter
    :param l2_reg: l2 regularization hyperparameter
    :return:
    """
    FLAGS.learning_rate = learning_rate
    FLAGS.keep_prob1 = keep_prob
    FLAGS.keep_prob2 = keep_prob
    FLAGS.momentum = momentum
    FLAGS.l2_reg = l2_reg


def set_other_flags(source_domain, source_year, target_domain, target_year):
    """
    Set other flags.

    :param source_domain: the source domain
    :param source_year: the year of the source domain dataset
    :param target_domain: the target domain
    :param target_year: the year of the target domain dataset
    :return:
    """
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.source_year = source_year
    FLAGS.target_year = target_year
    # FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
    #     FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.source_year) + "_BERT.txt"
    # FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
    #     FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + "_BERT.txt"
    FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_" + str(FLAGS.source_year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_" + str(FLAGS.target_year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    FLAGS.source_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_" + str(FLAGS.source_year) + "_BERT.txt"
    FLAGS.target_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_" + str(FLAGS.target_year) + "_BERT.txt"
    FLAGS.source_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.target_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    FLAGS.results_file = "data/programGeneratedData/RESULTS/" + str(FLAGS.embedding_dim) + "_threshold" + "_" + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + ".txt"

    FLAGS.results_file_nomask = "data/programGeneratedData/RESULTS/" + str(
        FLAGS.embedding_dim) + "_NoMask_" + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + ".txt"


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
