import numpy as np


def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .5}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'global_step': model.global_step,
    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    prediction = sess.run(model.prediction, feed_dict)
    acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return acc


def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)
