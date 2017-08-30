import tensorflow as tf
import pandas as pd
import numpy as np

names = ["class", "title", "content"]
MAX_LABEL = 15
# Small test data
# test_small_csv = pd.read_csv("./dbpedia_data/dbpedia_csv/test_small.csv", names=names)
# train_small_csv = pd.read_csv("./dbpedia_data/dbpedia_csv/train_small.csv", names=names)
#
# x_train = pd.Series(train_small_csv["content"])
# y_train = pd.Series(train_small_csv["class"])
# x_test = pd.Series(test_small_csv["content"])
# y_test = pd.Series(test_small_csv["class"])

# 70000  test  560000 train
test_csv = pd.read_csv("./dbpedia_data/dbpedia_csv/test.csv", names=names)
train_csv = pd.read_csv("./dbpedia_data/dbpedia_csv/train.csv", names=names)



shuffle_csv = train_csv.sample(frac=1)

frames = [shuffle_csv, test_csv]
concat_df = pd.concat(frames)

x_train = pd.Series(shuffle_csv["content"])
y_train = pd.Series(shuffle_csv["class"])
x_test = pd.Series(test_csv["content"])
y_test = pd.Series(test_csv["class"])

train_size = x_train.shape[0]
test_size = x_test.shape[0]

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.


# word_list = tf.unstack(word_vectors, axis=1)
# print(word_list) # shape=(345, 50)

# print(y_test)
def accuracy(pred, label):
    return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(label, 1))
            / pred.shape[0])



graph = tf.Graph()

batch_size = 128

with graph.as_default():

    # print("Test vectors: ")
    # print(test_vectors)

    # Input data
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE])
        y = tf.placeholder(tf.float32, [None, MAX_LABEL])

    # print(type(test_vectors))
    # test_dataset = tf.Variable(test_vectors, trainable=False)

    # Variables
    attn_weights = tf.Variable(tf.truncated_normal(([EMBEDDING_SIZE, 1])))


    # biases = tf.Variable(tf.zeros([MAX_LABEL, 1]))

    # Model
    def attn_model(data):
        # LSTM Cell
        word_list = tf.unstack(data, axis=1)
        with tf.name_scope('LSTMCell'):
            cell = tf.contrib.rnn.BasicLSTMCell(EMBEDDING_SIZE)
        outputs, states = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
        # print(encoding)

        print(len(outputs))  # MAX_DOCUMENT_SIZE * ( batch_size, EMBEDDING_SIZE)
        outputs = tf.stack(outputs)  # (MAX_DOCUMENT_SIZE, batch_size, EMBEDDING_SIZE)
        print(outputs.shape)
        outputs = tf.transpose(outputs, [1, 0, 2])
        print(outputs.shape)
        print(outputs.get_shape())
        shape = outputs.shape.as_list()
        dynamic_batch_size = tf.shape(outputs)[0]
        #  shape[0] : batch_size  shape[1]: MAX_DOCUMENT_SIZE  shape[2]: EMBEDDING_SIZE
        H = tf.reshape(outputs, [dynamic_batch_size * shape[1], shape[2]])  # H  ( batch_size * T, EMBEDDING_SIZE)
        M = tf.tanh(H)  # M  ( batch_size * T, EMBEDDING_SIZE)
        # with tf.name_scope('attention_vector'):
        print(attn_weights.shape)
        attention_vector = tf.nn.softmax(tf.matmul(M, attn_weights))
        attention_vector = tf.reshape(attention_vector, shape=(dynamic_batch_size, shape[1], 1))  # (batch_size, T)

        # H_Array = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
        # H_List = H_Array.unstack(outputs)
        # print(tf.shape(H_List))
        o_reshape = tf.reshape(outputs, shape=(dynamic_batch_size, EMBEDDING_SIZE, MAX_DOCUMENT_LENGTH))
        # print("Shape : ")
        # print(o_reshape.shape)
        # print(attention_vector.shape)
        r = tf.matmul(o_reshape, attention_vector)
        # print(r)
        R = tf.reshape(r, shape=(dynamic_batch_size, EMBEDDING_SIZE))
        # r = tf.matmul(tf.transpose(H), attention_vector) # (EMBEDDING_SIZE, 1)
        h_star = tf.tanh(R)
        print("h star shape:", h_star.shape)
        # , kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
        return tf.layers.dense(h_star, MAX_LABEL, activation=None)


    def lstm_model(data):
        word_list = tf.unstack(data, axis=1)
        cell = tf.contrib.rnn.BasicLSTMCell(EMBEDDING_SIZE)
        outputs, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
        outputs = tf.stack(outputs)  # (MAX_DOCUMENT_SIZE, batch_size, EMBEDDING_SIZE)
        print(outputs.shape)

        outputs = tf.transpose(outputs, [1, 0, 2])
        print(outputs.shape)  # (batch_size， MAX_DOCUMENT_SIZE, EMBEDDING_SIZE)

        return tf.layers.dense(encoding[1], MAX_LABEL, activation=None)


    # h_attn = attn_model(x)
    logits = attn_model(x)
    # logits = lstm_model(x)
    # print("Logits shape: ", logits.shape) # supposed to be [batch_size, MAX_LABELS]
    # print("Labels shape: ", y.shape) # [ batch_size, MAX_LABELS ]
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
               #\ + 0.01 * tf.nn.l2_loss(attn_weights)
        tf.summary.scalar('loss', loss)
    # + 0.001 * tf.nn.l2_loss(weights['attn_weights']) \
    #  + 0.001 * tf.nn.l2_loss(weights['out_weights'])

    # Opitimizer
    lr = 1
    with tf.name_scope("train_optimizer"):
        optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # Predictions
    train_prediction = tf.nn.softmax(logits)
    # test_prediction = tf.nn.softmax(attn_model(test_vec))


steps = 10000001

with tf.Session(graph=graph) as sess:
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    # Learn the vocabulary
    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test) # Generator

    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)


    x_train = np.array(x_train_list)
    print(x_train)
    x_test = np.array(x_test_list)
    n_words = len(vocab_processor.vocabulary_)

    print('Total words: %d' % n_words)
    # 345 * max document size
    # print(x_train.shape)

    merged_dataset = np.concatenate((x_train, x_test), axis=0)

    # labels [batch_size, MAX_LABEL]
    train_labels = tf.one_hot(y_train, MAX_LABEL, 1, 0)
    test_labels = tf.one_hot(y_test, MAX_LABEL, 1, 0)
    # print(merged_dataset)

    # word embedding together
    word_vectors = tf.contrib.layers.embed_sequence(
        merged_dataset, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    print(word_vectors.get_shape)

    #
    # test_vectors = tf.contrib.layers.embed_sequence(
    #     x_test, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)


    # merge
    merged = tf.summary.merge_all()
    # graph
    writer = tf.summary.FileWriter("tmp/logs/", sess.graph)


    tf.global_variables_initializer().run()
    print('Initilized')
    shape = word_vectors.get_shape()

    word_vectors_array = word_vectors.eval()
    word_labels = train_labels.eval()
    
    for step in range(steps):
        # print("In loop")
        offset = (step * batch_size) % (train_size - batch_size)
        batch_data = word_vectors_array[offset:offset + batch_size, :]

        batch_labels = word_labels[offset: (offset + batch_size), :]
        feed_dict = {
            x: batch_data,
            y: batch_labels
        }
        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            result = sess.run(merged, feed_dict=feed_dict)
            writer.add_summary(result, step)

            print("offest : ", offset)
            print("Minibatch loss at step %d : %f" % (step, l))
            # print(batch_data.shape, " ", batch_labels.shape)

            # print(predictions)
            print('Minibatch accuracy : %.1f %%' % accuracy(
                predictions, batch_labels))
            print()

    # print(word_vectors_array[:1000])
    # test_data = test_vectors.eval()
    # print(test_data.shape)
    # # print(test_data[:1000])
    test_labels = test_labels.eval()
    # print(test_labels.shape)
    # # offset = 0
    test_data = word_vectors_array[train_size: , :]
    print(test_data.shape, " ", test_labels.shape)
    test_dict = {
        x: test_data,
        y: test_labels
    }

    test_pred = sess.run([train_prediction], feed_dict=test_dict)
    test_size = len(test_labels)
    print(len(test_pred[0]))
    print(len(test_labels))

    cnt = 0
    for predict, real in zip(np.argmax(test_pred[0], 1), np.argmax(test_labels, 1)):
        if predict == real:
            cnt += 1
        # print(predict, ": ", real)

    print(cnt)
    print("Test accuracy:  %.1f %%" % (cnt / test_size * 100.0))

