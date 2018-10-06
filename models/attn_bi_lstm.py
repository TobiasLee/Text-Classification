from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from utils.prepare_data import *
import time
from utils.model_helper import *


class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph")
        # Word embedding
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)

        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))

        # prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


if __name__ == '__main__':
    # load data
    x_train, y_train = load_data("../dbpedia_data/dbpedia_csv/train.csv", sample_ratio=1e-2, one_hot=False)
    x_test, y_test = load_data("../dbpedia_data/dbpedia_csv/test.csv", one_hot=False)

    # data preprocessing
    x_train, x_test, vocab_size = \
        data_preprocessing_v2(x_train, x_test, max_len=32)
    print("train size: ", len(x_train))
    print("vocab size: ", vocab_size)

    # split dataset to test and dev
    x_test, x_dev, y_test, y_dev, dev_size, test_size = \
        split_dataset(x_test, y_test, 0.1)
    print("Validation Size: ", dev_size)

    config = {
        "max_len": 32,
        "hidden_size": 64,
        "vocab_size": vocab_size,
        "embedding_size": 128,
        "n_class": 15,
        "learning_rate": 1e-3,
        "batch_size": 4,
        "train_epoch": 20
    }

    classifier = ABLSTM(config)
    classifier.build_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dev_batch = (x_dev, y_dev)
    start = time.time()
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
            # plot the attention weight
            # print(np.reshape(attn, (config["batch_size"], config["max_len"])))
        t1 = time.time()

        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_acc = run_eval_step(classifier, sess, dev_batch)
        print("validation accuracy: %.3f " % dev_acc)

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc = run_eval_step(classifier, sess, (x_batch, y_batch))
        test_acc += acc
        cnt += 1

    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
