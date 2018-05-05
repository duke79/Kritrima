import random

import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "../model/model_%s_%s.ckpt"


def train_gate():
    InX = [[1, 1], [1, 0], [0, 1], [0, 0]]
    OutX = [1, 0, 0, 0]

    # x = tf.Variable([[2.0], [-2.0]], dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32, shape=[2, 1])
    w = tf.Variable([[2.0], [2.0]], dtype=tf.float32, name="w")
    b = tf.Variable([4.0], dtype=tf.float32, name="b")
    o = tf.add(tf.matmul(tf.transpose(x), w), b)
    oExpected = tf.placeholder(dtype=tf.float32, shape=[1])
    model = tf.global_variables_initializer()
    error = tf.square(tf.subtract(o[0][0], oExpected))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.002).minimize(error)
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        sess.run(model)
        for i in range(10000):
            if (0 == i % 1000):
                print(str(i) + ":")
            for gate_state in range(4):
                oO, trainO, errorO, wO, bO = sess.run([o, train, error, w, b],
                                                      feed_dict={
                                                          x: np.expand_dims(np.array(InX[gate_state]), axis=1),
                                                          oExpected: np.expand_dims(np.array(OutX[gate_state]), axis=0)
                                                      })
                if (0 == i % 1000):
                    print("o: " + str(oO[0][0]))
                    print("error: " + str(errorO[0]))
                    print("w: " + str(wO))
                    print("b: " + str(bO))
            if (0 == i % 1000):
                print("\n")

                var_name_list = [v.name for v in tf.trainable_variables()]
                randNum = random.randint(1, 100000)
                path = os.path.normpath(MODEL_PATH % (i, randNum))
                saver.save(sess, path)


def test_gate():
    w = tf.get_variable("w", shape=[2, 1])
    b = tf.get_variable("b", shape=[1])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, MODEL_PATH % (5000, 90092))
        print(w.eval())
        print(b.eval())


if __name__ == "__main__":
    # train_gate()
    test_gate()
