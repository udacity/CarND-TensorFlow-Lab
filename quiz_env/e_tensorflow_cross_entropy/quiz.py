import tensorflow as tf


def run():
    output = None
    softmax_data = [0.7, 0.2, 0.1]
    one_hot_encod_label = [1.0, 0.0, 0.0]

    softmax = tf.placeholder(tf.float32)
    one_hot_encod = tf.placeholder(tf.float32)

    cross_entropy = -tf.reduce_sum(one_hot_encod * tf.log(softmax))

    with tf.Session() as sess:
        output = sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot_encod: one_hot_encod_label})

    return output
