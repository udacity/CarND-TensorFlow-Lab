import tensorflow as tf


def run():
    """
    Return the number 123 using TensorFlow
    """
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123})  # Edit this line

    return output
