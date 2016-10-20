import tensorflow as tf


def run():
    output = None

    x = tf.Variable([1, 2, 3, 4])

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(x)

    return output
