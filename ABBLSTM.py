import tensorflow as tf
import pandas as pd
import numpy as np
from attention import attention
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time


names = ["class", "title", "content"]

test_csv = pd.read_csv("./dbpedia_data/dbpedia_csv/test.csv", names=names)
train_csv = pd.read_csv("./dbpedia_data/dbpedia_csv/train.csv", names=names)
MAX_LABEL = 15

shuffle_csv = train_csv.sample(frac=1)

x_train = pd.Series(shuffle_csv["content"])
y_train = pd.Series(shuffle_csv["class"])
x_test = pd.Series(test_csv["content"])
y_test = pd.Series(test_csv["class"])

# print(y_test)
train_size = x_train.shape[0]
test_size = x_test.shape[0]

dev_size = test_size // 100 * 1
test_size -= dev_size
print("Train size: ", train_size)
print("Dev size : ", dev_size)

# Hyperparameter
MAX_DOCUMENT_LENGTH = 25
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 64
ATTENTION_SIZE = 64
lr = 1e-3
BATCH_SIZE = 256
KEEP_PROB = 0.5
LAMBDA = 0.0001

MAX_LABEL = 15

# pre-processing
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
    MAX_DOCUMENT_LENGTH)
x_transform_train = vocab_processor.fit_transform(x_train)
x_transform_test = vocab_processor.transform(x_test)  # Generator

x_train_list = list(x_transform_train)
x_test_list = list(x_transform_test)

x_train = np.array(x_train_list)
# print(x_train)
x_test = np.array(x_test_list)
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

graph = tf.Graph()
with graph.as_default():

    batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.placeholder(tf.float32)

    embeddings_var = tf.Variable(tf.random_uniform([n_words, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_x)
    W = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1))
    # print(batch_embedded.shape)  # (?, 256, 100)

    rnn_outputs, _ = bi_rnn(BasicLSTMCell(HIDDEN_SIZE), BasicLSTMCell(HIDDEN_SIZE),
                            inputs=batch_embedded, dtype=tf.float32)
    # Attention
    fw_outputs = rnn_outputs[0]
    print(fw_outputs.shape)
    bw_outputs = rnn_outputs[1]
    H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
    M = tf.tanh(H) # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
    print(M.shape)
    # alpha (bs * sl, 1)
    alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, HIDDEN_SIZE]), tf.reshape(W, [-1, 1])))
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_DOCUMENT_LENGTH, 1])) # supposed to be (batch_size * HIDDEN_SIZE, 1)
    print(r.shape)
    r = tf.squeeze(r)
    h_star = tf.tanh(r) # (batch , HIDDEN_SIZE
    # attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)


    drop = tf.nn.dropout(h_star, keep_prob)
    # shape = drop.get_shape()
    # print(shape)

    # Fully connected layer（dense layer)
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, MAX_LABEL], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    # print(y_hat.shape)

    # y_hat = tf.squeeze(y_hat)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=batch_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))

steps = 10001 # about 5 epoch
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized! ")

    train_labels = tf.one_hot(y_train, MAX_LABEL, 1, 0)
    test_labels = tf.one_hot(y_test, MAX_LABEL, 1, 0)

    y_train = train_labels.eval()
    y_test = test_labels.eval()

    dev_x = x_test[: dev_size, :]
    dev_y = y_test[: dev_size, :]

    test_x = x_test[dev_size:, :]
    test_y = y_test[dev_size:, :]

    offset = 0
    print("Start trainning")
    start = time.time()
    for step in range(steps):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = x_train[offset: offset + BATCH_SIZE, :]
        batch_label = y_train[offset: offset + BATCH_SIZE, :]
        # print(batch_x.shape)
        # print(batch_y.shape)
        fd = {batch_x: batch_data, batch_y: batch_label, keep_prob: KEEP_PROB}
        l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

        if step % 100 == 0:
            print()
            print("Step %d: loss : %f   accuracy: %f %%" % (step, l, acc * 100))

        if step % 500 == 0:
            print("******************************\n")
            dev_loss, dev_acc = sess.run([loss, accuracy], feed_dict={batch_x: dev_x, batch_y: dev_y, keep_prob: 1})
            print("Dev set at Step %d: loss : %f   accuracy: %f %%\n" % (step, dev_loss, dev_acc * 100))
            print("******************************")

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("start predicting:  \n")
    test_accuracy = sess.run([accuracy], feed_dict={batch_x: test_x, batch_y: test_y, keep_prob: 1})
    print("Test accuracy : %f %%", test_accuracy[0]*100)




