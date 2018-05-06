import random

import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "../model/model_%s_%s.ckpt"
TB_PATH = "../logs/1/train"
EPOCH = 10000
LEARNING_RATE = 0.02
ENABLE_TENSORBOARD = False


def train_gate():
    InX = [[1, 1], [1, 0], [0, 1], [0, 0]]
    OutX = [1, 0, 0, 0]

    x = tf.placeholder(dtype=tf.float32, shape=[2, 1])

    w1 = tf.get_variable("w1", dtype=tf.float32, shape=[2, 1], initializer=tf.constant_initializer([[1.0], [2.0]]))
    b1 = tf.get_variable("b1", dtype=tf.float32, shape=[2, 1], initializer=tf.constant_initializer([2.0]))
    o1 = tf.add(tf.matmul(tf.transpose(x), w1), b1)

    w2 = tf.get_variable("w2", dtype=tf.float32, shape=[2, 1], initializer=tf.constant_initializer([[3.0], [8.0]]))
    b2 = tf.get_variable("b2", dtype=tf.float32, shape=[2, 1], initializer=tf.constant_initializer([14.0]))
    o2 = tf.add(tf.matmul(tf.transpose(x), w2), b2)

    w = tf.get_variable("w", dtype=tf.float32, shape=[2, 1], initializer=tf.constant_initializer([[2.0], [1.0]]))
    b = tf.get_variable("b", dtype=tf.float32, shape=[2, 1], initializer=tf.constant_initializer([-4.0]))
    oRaw = tf.sigmoid(tf.add(tf.add(tf.multiply(w[0], o1), tf.multiply(w[1], o2)), b))
    o = tf.cond(oRaw[0][0] > 0, lambda: tf.divide(oRaw[0][0], oRaw[0][0]), lambda: tf.subtract(oRaw[0][0], oRaw[0][0]))

    oExpected = tf.placeholder(dtype=tf.float32, shape=[1])
    # diff = tf.subtract(o, oExpected)
    # error = tf.add(tf.square(diff), tf.sqrt(tf.abs(diff)))
    # train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
    error = tf.square(tf.subtract(oRaw[0][0], oExpected))
    train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
    saver = tf.train.Saver(max_to_keep=10)

    if ENABLE_TENSORBOARD:
        tf.summary.histogram("w1", w1)
        tf.summary.histogram("b1", b1)
        tf.summary.histogram("w2", w2)
        tf.summary.histogram("b2", b2)
        tf.summary.histogram("w", w)
        tf.summary.histogram("b", b)

    model = tf.global_variables_initializer()

    with tf.Session() as sess:

        tb_writer = None
        if ENABLE_TENSORBOARD:
            tb_writer = tf.summary.FileWriter(TB_PATH, sess.graph)

        sess.run(model)
        w2O = sess.run(w2)
        print(w2O)
        for i in range(EPOCH):
            errorTotal = 0.0
            # if (0 == i % (EPOCH / 10)):
            #     print(str(i) + ":")
            for gate_state in range(4):
                tb_merge = tf.summary.merge_all()

                trainO, oO, errorO, wO, bO, xO \
                    = sess.run(
                    [train, oRaw, error, w2, b2, x],
                    feed_dict={
                        x: np.expand_dims(np.array(InX[gate_state]),
                                          axis=1),
                        oExpected: np.expand_dims(
                            np.array(OutX[gate_state]),
                            axis=0)
                    })

                tb_summary = None
                if ENABLE_TENSORBOARD:
                    tb_summary = sess.run(tb_merge)

                if ENABLE_TENSORBOARD:
                    tb_writer.add_summary(tb_summary, gate_state)

                if (0 == i % (EPOCH / 10)):
                    print("x: " + str(xO))
                    print("o: " + str(oO[0][0]))
                    # print("error: " + str(errorO[0]))
                    errorTotal += errorO[0]
                    # print("w: " + str(wO))
                    # print("b: " + str(bO))
                    print("\n")
            if 0 == i % (EPOCH / 10):
                print("error: " + str(errorTotal))
                print("\n")

                randNum = random.randint(1, 100000)
                path = os.path.normpath(MODEL_PATH % (i, randNum))
                saver.save(sess, path)


def test_gate():
    w = tf.get_variable("w", shape=[2, 1])
    b = tf.get_variable("b", shape=[1])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, MODEL_PATH % (9000, 49074))
        print(w.eval())
        print(b.eval())


if __name__ == "__main__":
    train_gate()
    # test_gate()
