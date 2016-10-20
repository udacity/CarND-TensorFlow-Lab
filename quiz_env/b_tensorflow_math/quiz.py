import tensorflow as tf


def run():
    output = None
    x = tf.constant(10)
    y = tf.constant(2)
    z = tf.div(x, y) - 1

    with tf.Session() as sess:
        output = sess.run(z)

    return output
