import tensorflow as tf


def run():
    output = None
    x_data = [[1.0, 3.0, 2.0], [2.5, 2.0, 6.3]]
    weights = [[-0.3545495, -0.17928936], [-0.63093454, 0.74906588], [0.74592733, -0.04424516]]
    class_size = 2

    x = tf.placeholder(tf.float32)
    biases = tf.Variable(tf.zeros([class_size]))

    logits = tf.matmul(x, weights) + biases

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(logits, feed_dict={x: x_data})

    return output
