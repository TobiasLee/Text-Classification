import time

from modules.multihead import *
from utils.prepare_data import *

# Hyperparameter
MAX_DOCUMENT_LENGTH = 25
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
ATTENTION_SIZE = 64
lr = 1e-3
BATCH_SIZE = 1024
KEEP_PROB = 0.5
LAMBDA = 1e-3
MAX_LABEL = 15
epochs = 15

# load data
x_train, y_train = load_data("../dbpedia_csv/train.csv", sample_ratio=1)
x_test, y_test = load_data("../dbpedia_csv/test.csv", sample_ratio=1)

# data preprocessing
x_train, x_test, vocab, vocab_size = \
    data_preprocessing(x_train, x_test, MAX_DOCUMENT_LENGTH)
print(vocab_size)

# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 0.1)
print("Validation size: ", dev_size)

graph = tf.Graph()
with graph.as_default():

    batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.placeholder(tf.float32)

    embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_x)
    # multihead attention
    outputs = multihead_attention(queries=batch_embedded, keys=batch_embedded)
    # FFN(x) = LN(x + point-wisely NN(x))
    outputs = feedforward(outputs, [HIDDEN_SIZE, EMBEDDING_SIZE])
    print(outputs.shape)
    outputs = tf.reshape(outputs, [-1, MAX_DOCUMENT_LENGTH * EMBEDDING_SIZE])
    logits = tf.layers.dense(outputs, units=MAX_LABEL)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(tf.nn.softmax(logits), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))

steps = 10001 # about 5 epoch
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized! ")

    print("Start trainning")
    start = time.time()
    for e in range(epochs):

        epoch_start = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

        epoch_finish = time.time()
        print("Validation accuracy and loss: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_dev,
            batch_y: y_dev,
            keep_prob: 1.0
        }))
        print("epoch time:", epoch_finish - epoch_start , " s")

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: 1.0}
            acc = sess.run(accuracy, feed_dict=fd)
            test_acc += acc
            cnt += 1        
    
    print("Test accuracy : %f %%" % ( test_acc / cnt * 100))




