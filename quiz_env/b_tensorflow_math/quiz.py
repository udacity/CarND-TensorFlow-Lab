import tensorflow as tf


def run():
    """
    Return the TensorFlow session output of 10/2 - 1
    """
    output = None
    x = tf.constant(10)
    y = tf.constant(2)
    z = tf.div(x, y) - 1

    with tf.Session() as sess:
        output = sess.run(z)

    return output
