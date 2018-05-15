import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
from utils.prepare_data import *

# Hyperparameters
MAX_DOCUMENT_LENGTH = 25
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 64
BATCH_SIZE = 256
KEEP_PROB = 0.5
epsilon = 5.0  # IMDB ideal norm length
MAX_LABEL = 15
epochs = 10

# load data
x_train, y_train = load_data("../dbpedia_data/dbpedia_csv/train.csv", sample_ratio=0.01)
x_test, y_test = load_data("../dbpedia_data/dbpedia_csv/test.csv", sample_ratio=0.1)

# data preprocessing
x_train, x_test, vocab, vocab_size = \
    data_preprocessing(x_train, x_test, MAX_DOCUMENT_LENGTH)
print("Vocab size: ", vocab_size)

# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 0.1)
print("Validation size: ", dev_size)


def get_freq(vocabulary):
    vocab_freq = vocabulary._freq
    words = vocab_freq.keys()
    freq = [0] * vocab_size
    for word in words:
        word_idx = vocab.get(word)
        word_freq = vocab_freq[word]
        freq[word_idx] = word_freq

    return freq


def _scale_l2(x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def add_perturbation(embedded, loss):
    """Adds gradient to embedding and recomputes classification loss."""
    grad, = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_l2(grad, epsilon)
    return embedded + perturb


def normalize(emb, weights):
    # weights = vocab_freqs / tf.reduce_sum(vocab_freqs) ?? 这个实现没问题吗
    print("Weights: ", weights)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev


graph = tf.Graph()
with graph.as_default():
    batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.placeholder(tf.float32)
    vocab_freqs = tf.constant(get_freq(vocab), dtype=tf.float32, shape=(vocab_size, 1))

    weights = vocab_freqs / tf.reduce_sum(vocab_freqs)

    embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    W = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1))
    W_fc = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, MAX_LABEL], stddev=0.1))
    b_fc = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))

    embedding_norm = normalize(embeddings_var, weights)
    batch_embedded = tf.nn.embedding_lookup(embedding_norm, batch_x)


    def cal_loss_logit(batch_embedded, keep_prob, reuse=True, scope="loss"):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            rnn_outputs, _ = bi_rnn(BasicLSTMCell(HIDDEN_SIZE), BasicLSTMCell(HIDDEN_SIZE),
                                    inputs=batch_embedded, dtype=tf.float32)

            # Attention

            H = tf.add(rnn_outputs[0], rnn_outputs[1])  # fw + bw
            M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
            print(M.shape)
            # alpha (bs * sl, 1)
            alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, HIDDEN_SIZE]), tf.reshape(W, [-1, 1])))
            r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_DOCUMENT_LENGTH,
                                                                         1]))  # supposed to be (batch_size * HIDDEN_SIZE, 1)
            print(r.shape)
            r = tf.squeeze(r)
            h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE
            # attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
            drop = tf.nn.dropout(h_star, keep_prob)

            # Fully connected layer（dense layer)

            y_hat = tf.nn.xw_plus_b(drop, W_fc, b_fc)

        return y_hat, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=batch_y))


    lr = 1e-3
    logits, cl_loss = cal_loss_logit(batch_embedded, keep_prob, reuse=False)
    embedding_perturbated = add_perturbation(batch_embedded, cl_loss)
    ad_logits, ad_loss = cal_loss_logit(embedding_perturbated, keep_prob, reuse=True)
    loss = cl_loss + ad_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(tf.nn.softmax(logits), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized! ")

    print("Start training")
    start = time.time()
    time_consumed = 0
    for e in range(epochs):

        epoch_start = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

        epoch_finish = time.time()
        print("Validation accuracy: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_dev,
            batch_y: y_dev,
            keep_prob: 1.0
        }))

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("start predicting:  \n")
    test_accuracy = sess.run([accuracy], feed_dict={batch_x: x_test, batch_y: y_test, keep_prob: 1})
    print("Test accuracy : %f %%" % (test_accuracy[0] * 100))
