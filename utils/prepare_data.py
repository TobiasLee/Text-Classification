import numpy as np
import pandas as pd
import tensorflow as tf

names = ["class", "title", "content"]


def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, sample_ratio= 1, n_class=15, names=names):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    y = pd.Series(shuffle_csv["class"])
    y = to_one_hot(y, n_class)
    print(y.shape)
    return x, y


def data_preprocessing(train, test, max_len):
    '''transform to one-hot idx vector by VocabularyProcess'''
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab, vocab_size


def split_dataset(x_test, y_test, dev_ratio):
    '''split test dataset to test and dev set with ratio '''
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    '''Generator to yield batches'''
    # Shuffle data first.
    perm = np.random.permutation(data_X.shape[0])
    data_X = data_X[perm]
    data_Y = data_Y[perm]
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch
