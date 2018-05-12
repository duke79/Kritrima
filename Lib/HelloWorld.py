import random

import tensorflow as tf
import numpy as np
import os


def train_gate(TEST=False,
               EPOCH=10000,
               LEARNING_RATE=0.02,
               BATCH_SIZE=4,
               SHAPE=[2, 1],
               InX=[[1, 1], [1, 0], [0, 1], [0, 0]],
               OutX=[1, 0, 0, 0],
               ENABLE_TENSORBOARD=False,
               MODEL_PATH_ARGS=(4000, 80816)):
    MODEL_PATH = "../model/model_%s_%s.ckpt"
    TB_PATH = "../logs/1/train"

    x = tf.placeholder(dtype=tf.float32, shape=SHAPE)

    w1 = tf.get_variable("w1", dtype=tf.float32, shape=SHAPE, initializer=tf.constant_initializer([[1.0], [2.0]]))
    b1 = tf.get_variable("b1", dtype=tf.float32, shape=SHAPE, initializer=tf.constant_initializer([2.0]))
    o1 = tf.add(tf.matmul(tf.transpose(x), w1), b1)

    w2 = tf.get_variable("w2", dtype=tf.float32, shape=SHAPE, initializer=tf.constant_initializer([[3.0], [8.0]]))
    b2 = tf.get_variable("b2", dtype=tf.float32, shape=SHAPE, initializer=tf.constant_initializer([14.0]))
    o2 = tf.add(tf.matmul(tf.transpose(x), w2), b2)

    w = tf.get_variable("w", dtype=tf.float32, shape=SHAPE, initializer=tf.constant_initializer([[2.0], [1.0]]))
    b = tf.get_variable("b", dtype=tf.float32, shape=SHAPE, initializer=tf.constant_initializer([-4.0]))
    # oRaw = tf.sigmoid(tf.add(tf.add(tf.multiply(w[0], o1), tf.multiply(w[1], o2)), b))
    oRaw = tf.sigmoid(tf.add(tf.matmul(tf.transpose(x), w1), b1))
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
        if TEST:
            saver.restore(sess, MODEL_PATH % (MODEL_PATH_ARGS[0], MODEL_PATH_ARGS[1]))
            print("w1" + str(w1.eval()))
            print("b1" + str(b1.eval()))
            print("w2" + str(w2.eval()))
            print("b2" + str(b2.eval()))
            print("w" + str(w.eval()))
            print("b" + str(b.eval()))
            print("\n")
        else:
            sess.run(model)

        tb_writer = None
        if ENABLE_TENSORBOARD:
            tb_writer = tf.summary.FileWriter(TB_PATH, sess.graph)

        if TEST:
            EPOCH = 1

        for i in range(EPOCH):
            errorTotal = 0.0
            for batch_itr in range(BATCH_SIZE):
                tb_merge = None
                if TEST:
                    oO, errorO, wO, bO, oO2, xO \
                        = sess.run(
                        [oRaw, error, w2, b2, o2, x],
                        feed_dict={
                            x: np.expand_dims(np.array(InX[batch_itr]),
                                              axis=1),
                            oExpected: np.expand_dims(
                                np.array(OutX[batch_itr]),
                                axis=0)
                        })
                else:
                    tb_merge = tf.summary.merge_all()

                    trainO, oO, errorO, wO, bO, oO2, xO \
                        = sess.run(
                        [train, oRaw, error, w2, b2, o2, x],
                        feed_dict={
                            x: np.expand_dims(np.array(InX[batch_itr]),
                                              axis=1),
                            oExpected: np.expand_dims(
                                np.array(OutX[batch_itr]),
                                axis=0)
                        })

                tb_summary = None
                if ENABLE_TENSORBOARD:
                    tb_summary = sess.run(tb_merge)

                if ENABLE_TENSORBOARD:
                    tb_writer.add_summary(tb_summary, batch_itr)

                if 0 == i % (EPOCH / 10):
                    print("x: " + str(xO))
                    # print("o2: " + str(oO2))
                    print("o: " + str(oO[0][0]))
                    # print("error: " + str(errorO[0]))
                    errorTotal += errorO[0]
                    # print("w: " + str(wO))
                    # print("b: " + str(bO))
                    print("\n")
            if 0 == i % (EPOCH / 10):
                print("error: " + str(errorTotal))
                print("\n")
                if not TEST:
                    randNum = random.randint(1, 100000)
                    path = os.path.normpath(MODEL_PATH % (i, randNum))
                    saver.save(sess, path)


if __name__ == "__main__":
    train_gate(SHAPE=[3, 1],
               InX=[[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]],
               OutX=[1, 0, 0, 0])
    # train_gate(TEST=True, MODEL_PATH_ARGS=(4000, 80816))
